import json
import uuid
import copy
from datetime import datetime
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
import httpx
from httpx import HTTPStatusError
from loguru import logger
from pydantic import BaseModel, ValidationError

from api.constants import DEPLOYMENT_MODE
from api.db import db_client
from api.db.models import UserModel
from api.db.workflow_template_client import WorkflowTemplateClient
from api.enums import CallType
from api.schemas.workflow import WorkflowRunResponseSchema
from api.services.auth.depends import get_user
from api.services.mps_service_key_client import mps_service_key_client
from api.services.workflow.dto import ReactFlowDTO
from api.services.workflow.errors import ItemKind, WorkflowError
from api.services.workflow.workflow import WorkflowGraph
from api.services.gen_ai.json_parser import parse_llm_json


async def _generate_workflow_locally(request, user: UserModel) -> dict:
    """Generate a workflow JSON using the user's configured LLM."""
    
    # 1. Get User LLM Config
    user_config = await db_client.get_user_configurations(user.id)
    if not user_config or not user_config.llm or not user_config.llm.api_key:
         logger.warning("No LLM configuration found for local generation.")
         return None

    llm_config = user_config.llm
    provider = llm_config.provider
    api_key = llm_config.api_key
    model = llm_config.model or "gpt-4o"

    # 2. Prepare Prompt
    system_prompt = """You are an expert voice agent designer. Create a JSON workflow based on the user's request.
    
    The JSON structure must strictly follow this schema:
    {
      "nodes": [
        {
          "id": "trigger_1",
          "type": "trigger", 
          "position": {"x": 100, "y": 100},
          "data": { "name": "Start Call", "trigger_path": "<UUID>", "type": "inbound" }
        },
        {
          "id": "node_2",
          "type": "agentNode",
          "position": {"x": 100, "y": 300},
          "data": {
            "name": "Greeting",
            "prompt": "Say hello to the user.",
            "is_start": true,
            "wait_for_user_response": true
          }
        },
        {
            "id": "node_3",
            "type": "endCall",
            "position": { "x": 100, "y": 500 },
            "data": { "name": "End Call" }
        }
      ],
      "edges": [
        { 
            "id": "e1-2", 
            "source": "trigger_1", 
            "target": "node_2", 
            "data": { "label": "Start", "condition": "always" } 
        },
        {
            "id": "e2-3",
            "source": "node_2",
            "target": "node_3",
            "data": { "label": "End", "condition": "always" }
        }
      ]
    }

    Rules:
    1. Node 'type' MUST be one of: 'trigger', 'agentNode', 'startCall', 'endCall', 'webhook'.
    2. Node 'data' MUST have a 'name' field string (min length 1). DO NOT use 'label'.
    3. Edge 'data' MUST have 'label' AND 'condition' fields (strings, min length 1).
    4. Always start with a 'trigger' node.
    5. Connect trigger to an 'agentNode' or 'startCall'.
    6. Use 'agentNode' for speaking/listening with the user.
    7. Use 'endCall' to hang up.
    8. 'nodes' and 'edges' lists are required.
    9. Ensure all node IDs are unique strings.
    10. 'trigger_path' in trigger node should be a UUID.
    11. For 'agentNode', 'prompt' field in data is REQUIRED.
    """

    user_prompt = f"Use Case: {request.use_case}\nDescription: {request.activity_description}\nCall Type: {request.call_type}"

    # 3. Call LLM API (Generic OpenAI-compatible)
    base_url = None
    if provider == "openai":
        base_url = "https://api.openai.com/v1"
    elif provider == "groq":
        base_url = "https://api.groq.com/openai/v1"
    elif provider == "openrouter":
        base_url = "https://openrouter.ai/api/v1"
    elif provider == "deepseek":
        base_url = "https://api.deepseek.com/v1"
    
    if not base_url:
        logger.warning(f"Provider {provider} not supported for local generation.")
        return None

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.7,
                    "response_format": {"type": "json_object"}
                }
            )
            
            if response.status_code != 200:
                logger.error(f"LLM API Error: {response.text}")
                return None
                
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # 4. Parse JSON
            workflow_def = parse_llm_json(content)
            
            if not workflow_def or "nodes" not in workflow_def:
                return None

            return {
                "name": request.use_case,
                "workflow_definition": workflow_def
            }

    except Exception as e:
        logger.error(f"Local workflow generation failed: {e}")
        return None



def extract_trigger_paths(workflow_definition: dict) -> List[str]:
    """Extract trigger UUIDs from workflow definition.

    Args:
        workflow_definition: The workflow definition JSON

    Returns:
        List of trigger UUIDs found in the workflow
    """
    if not workflow_definition:
        return []

    nodes = workflow_definition.get("nodes", [])
    trigger_paths = []

    for node in nodes:
        if node.get("type") == "trigger":
            trigger_path = node.get("data", {}).get("trigger_path")
            if trigger_path:
                trigger_paths.append(trigger_path)

    return trigger_paths


def regenerate_trigger_uuids(workflow_definition: dict) -> dict:
    """Regenerate UUIDs for all trigger nodes in a workflow definition.

    This should be called when creating a new workflow from a template or
    duplicating a workflow to avoid trigger UUID conflicts.

    Args:
        workflow_definition: The workflow definition JSON

    Returns:
        Updated workflow definition with new trigger UUIDs
    """
    if not workflow_definition:
        return workflow_definition

    # Deep copy to avoid modifying the original
    import copy

    updated_definition = copy.deepcopy(workflow_definition)

    nodes = updated_definition.get("nodes", [])
    for node in nodes:
        if node.get("type") == "trigger":
            # Generate a new UUID for this trigger
            if "data" not in node:
                node["data"] = {}
            node["data"]["trigger_path"] = str(uuid.uuid4())

    return updated_definition


router = APIRouter(prefix="/workflow")


class ValidateWorkflowResponse(BaseModel):
    is_valid: bool
    errors: list[WorkflowError]


class WorkflowResponse(BaseModel):
    id: int
    name: str
    status: str
    created_at: datetime
    workflow_definition: dict
    current_definition_id: int | None
    template_context_variables: dict | None = None
    call_disposition_codes: dict | None = None
    total_runs: int | None = None
    workflow_configurations: dict | None = None


class WorkflowListResponse(BaseModel):
    """Lightweight response for workflow listings (excludes large fields)."""

    id: int
    name: str
    status: str
    created_at: datetime
    total_runs: int


class WorkflowCountResponse(BaseModel):
    """Response for workflow count endpoint."""

    total: int
    active: int
    archived: int


class WorkflowTemplateResponse(BaseModel):
    id: int
    template_name: str
    template_description: str
    template_json: dict
    created_at: datetime


class CreateWorkflowRequest(BaseModel):
    name: str
    workflow_definition: dict


class DuplicateTemplateRequest(BaseModel):
    template_id: int
    workflow_name: str


class UpdateWorkflowRequest(BaseModel):
    name: str
    workflow_definition: dict | None = None
    template_context_variables: dict | None = None
    workflow_configurations: dict | None = None


class UpdateWorkflowStatusRequest(BaseModel):
    status: str  # "active" or "archived"


class CreateWorkflowRunRequest(BaseModel):
    mode: str
    name: str


class CreateWorkflowRunResponse(BaseModel):
    id: int
    workflow_id: int
    name: str
    mode: str
    created_at: datetime
    definition_id: int
    initial_context: dict | None = None


class CreateWorkflowTemplateRequest(BaseModel):
    call_type: Literal[CallType.INBOUND.value, CallType.OUTBOUND.value]
    use_case: str
    activity_description: str


@router.post("/{workflow_id}/validate")
async def validate_workflow(
    workflow_id: int,
    user: UserModel = Depends(get_user),
) -> ValidateWorkflowResponse:
    """
    Validate all nodes in a workflow to ensure they have required fields.

    Args:
        workflow_id: The ID of the workflow to validate
        user: The authenticated user

    Returns:
        Object indicating if workflow is valid and any invalid nodes/edges
    """
    workflow = await db_client.get_workflow(
        workflow_id, organization_id=user.selected_organization_id
    )

    if workflow is None:
        raise HTTPException(
            status_code=404, detail=f"Workflow with id {workflow_id} not found"
        )

    errors: list[WorkflowError] = []

    # Get workflow definition from WorkflowDefinition table, fallback to workflow_definition field
    workflow_definition = workflow.workflow_definition_with_fallback

    # ----------- DTO Validation ------------
    dto: Optional[ReactFlowDTO] = None

    try:
        dto = ReactFlowDTO.model_validate(workflow_definition)
    except ValidationError as exc:
        errors.extend(_transform_schema_errors(exc, workflow_definition))

    # ----------- Graph Validation if DTO is valid ------------
    try:
        if dto:
            WorkflowGraph(dto)
    except ValueError as e:
        errors.extend(e.args[0])

    if errors:
        raise HTTPException(
            status_code=422,
            detail=ValidateWorkflowResponse(is_valid=False, errors=errors).model_dump(),
        )

    return ValidateWorkflowResponse(is_valid=True, errors=[])


def _transform_schema_errors(
    exc: ValidationError, workflow_definition: dict
) -> list[WorkflowError]:
    out: list[WorkflowError] = []

    for err in exc.errors():
        loc = err["loc"]
        idx = workflow_definition[loc[0]][loc[1]]["id"]

        kind: ItemKind = ItemKind.node if loc[0] == "nodes" else ItemKind.edge

        out.append(
            WorkflowError(
                kind=kind,
                id=idx,
                field=".".join(str(p) for p in err["loc"][2:]) or None,
                message=err["msg"].capitalize(),
            )
        )
    return out


@router.post("/create/definition")
async def create_workflow(
    request: CreateWorkflowRequest, user: UserModel = Depends(get_user)
) -> WorkflowResponse:
    """
    Create a new workflow from the client

    Args:
        request: The create workflow request
        user: The user to create the workflow for
    """
    workflow = await db_client.create_workflow(
        request.name,
        request.workflow_definition,
        user.id,
        user.selected_organization_id,
    )

    # Sync agent triggers if workflow definition contains any
    if request.workflow_definition:
        trigger_paths = extract_trigger_paths(request.workflow_definition)
        if trigger_paths:
            await db_client.sync_triggers_for_workflow(
                workflow_id=workflow.id,
                organization_id=user.selected_organization_id,
                trigger_paths=trigger_paths,
            )

    return {
        "id": workflow.id,
        "name": workflow.name,
        "status": workflow.status,
        "created_at": workflow.created_at,
        "workflow_definition": workflow.workflow_definition_with_fallback,
        "current_definition_id": workflow.current_definition_id,
        "template_context_variables": workflow.template_context_variables,
        "call_disposition_codes": workflow.call_disposition_codes,
        "workflow_configurations": workflow.workflow_configurations,
    }


@router.post("/create/template")
async def create_workflow_from_template(
    request: CreateWorkflowTemplateRequest,
    user: UserModel = Depends(get_user),
) -> WorkflowResponse:
    """
    Create a new workflow from a natural language template request.

    This endpoint:
    1. Uses mps_service_key_client to call MPS workflow API
    2. Passes organization ID (authenticated mode) or created_by (OSS mode)
    3. Creates the workflow in the database

    Args:
        request: The template creation request with call_type, use_case, and activity_description
        user: The authenticated user

    Returns:
        The created workflow

    Raises:
        HTTPException: If MPS API call fails
    """
    try:
        # Call MPS API to generate workflow using the client
        if DEPLOYMENT_MODE == "oss":
            workflow_data = await mps_service_key_client.call_workflow_api(
                call_type=request.call_type.upper(),
                use_case=request.use_case,
                activity_description=request.activity_description,
                created_by=str(user.provider_id),
            )
        else:
            if not user.selected_organization_id:
                raise HTTPException(status_code=400, detail="No organization selected")

            workflow_data = await mps_service_key_client.call_workflow_api(
                call_type=request.call_type.upper(),
                use_case=request.use_case,
                activity_description=request.activity_description,
                organization_id=user.selected_organization_id,
            )

    except HTTPStatusError as e:
        if e.response.status_code in [401, 403]:
            # Fallback for local users without creating a service key:
            logger.warning(f"MPS API unauthorized ({e.response.status_code}), attempting local generation")
            
            # 1. Try to generate using local LLM
            workflow_data = await _generate_workflow_locally(request, user)
            
            if not workflow_data:
                # 2. If local generation fails (no key or error), fallback to basic hardcoded workflow
                logger.warning("Local generation failed or not configured, falling back to basic workflow")
                trigger_uuid = str(uuid.uuid4())
                workflow_data = {
                    "name": f"{request.use_case}",
                    "workflow_definition": {
                        "nodes": [
                            {
                                "id": "trigger_1",
                                "type": "trigger",
                                "position": {"x": 250, "y": 50},
                                "data": {
                                    "name": "Start Call", 
                                    "trigger_path": trigger_uuid,
                                    "type": request.call_type
                                }
                            },
                            {
                                "id": "llm_1",
                                "type": "agentNode",
                                "position": {"x": 250, "y": 200},
                                "data": {
                                    "name": "AI Assistant",
                                    "prompt": request.activity_description or "You are a helpful assistant.",
                                    "is_start": True,
                                    "wait_for_user_response": True
                                }
                            },
                            {
                                "id": "hangup_1",
                                "type": "endCall",
                                "position": {"x": 250, "y": 400},
                                "data": {"name": "End Call"}
                            }
                        ],
                        "edges": [
                            {
                                "id": "e1-2",
                                "source": "trigger_1",
                                "target": "llm_1",
                                "sourceHandle": "output",
                                "targetHandle": "input",
                                "data": {"label": "Start", "condition": "always"}
                            },
                            {
                                "id": "e2-3",
                                "source": "llm_1",
                                "target": "hangup_1",
                                "sourceHandle": "output",
                                "targetHandle": "input",
                                "data": {"label": "End", "condition": "always"}
                            }
                        ]
                    }
                }
        else:
            logger.error(f"MPS API error: {e}")
            raise HTTPException(
                status_code=e.response.status_code if hasattr(e, "response") else 500,
                detail=str(e),
            )
            
    except Exception as e:
        logger.error(f"Unexpected error creating workflow from template: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}",
        )

    # Create the workflow in our database
    # Regenerate trigger UUIDs to avoid conflicts with existing triggers (if any)
    workflow_def = regenerate_trigger_uuids(
        workflow_data.get("workflow_definition", {})
    )
    
    workflow = await db_client.create_workflow(
        name=workflow_data.get("name", f"{request.use_case}"),
        workflow_definition=workflow_def,
        user_id=user.id,
        organization_id=user.selected_organization_id,
    )

    # Sync agent triggers if workflow definition contains any
    if workflow_def:
        trigger_paths = extract_trigger_paths(workflow_def)
        if trigger_paths:
            await db_client.sync_triggers_for_workflow(
                workflow_id=workflow.id,
                organization_id=user.selected_organization_id,
                trigger_paths=trigger_paths,
            )

    return {
        "id": workflow.id,
        "name": workflow.name,
        "status": workflow.status,
        "created_at": workflow.created_at,
        "workflow_definition": workflow.workflow_definition_with_fallback,
        "current_definition_id": workflow.current_definition_id,
        "template_context_variables": workflow.template_context_variables,
        "call_disposition_codes": workflow.call_disposition_codes,
        "workflow_configurations": workflow.workflow_configurations,
    }


class WorkflowSummaryResponse(BaseModel):
    id: int
    name: str


@router.get("/count")
async def get_workflow_count(
    user: UserModel = Depends(get_user),
) -> WorkflowCountResponse:
    """Get workflow counts for the authenticated user's organization.

    This is a lightweight endpoint for checking if the user has workflows,
    useful for redirect logic without fetching full workflow data.
    """
    counts = await db_client.get_workflow_counts(
        organization_id=user.selected_organization_id
    )

    return WorkflowCountResponse(
        total=counts["total"],
        active=counts["active"],
        archived=counts["archived"],
    )


@router.get("/fetch")
async def get_workflows(
    user: UserModel = Depends(get_user),
    status: Optional[str] = Query(
        None,
        description="Filter by status - can be single value (active/archived) or comma-separated (active,archived)",
    ),
) -> List[WorkflowListResponse]:
    """Get all workflows for the authenticated user's organization.

    Returns a lightweight response with only essential fields for listing.
    Use GET /workflow/fetch/{workflow_id} to get full workflow details.
    """
    # Handle comma-separated status values
    if status and "," in status:
        # Split comma-separated values and fetch workflows for each status
        status_list = [s.strip() for s in status.split(",")]
        all_workflows = []
        for status_value in status_list:
            workflows = await db_client.get_all_workflows_for_listing(
                organization_id=user.selected_organization_id, status=status_value
            )
            all_workflows.extend(workflows)
        workflows = all_workflows
    else:
        # Single status or no status filter
        workflows = await db_client.get_all_workflows_for_listing(
            organization_id=user.selected_organization_id, status=status
        )

    # Get run counts for all workflows in a single query
    workflow_ids = [workflow.id for workflow in workflows]
    run_counts = await db_client.get_workflow_run_counts(workflow_ids)

    return [
        WorkflowListResponse(
            id=workflow.id,
            name=workflow.name,
            status=workflow.status,
            created_at=workflow.created_at,
            total_runs=run_counts.get(workflow.id, 0),
        )
        for workflow in workflows
    ]


@router.get("/fetch/{workflow_id}")
async def get_workflow(
    workflow_id: int,
    user: UserModel = Depends(get_user),
) -> WorkflowResponse:
    """Get a single workflow by ID"""
    workflow = await db_client.get_workflow(
        workflow_id, organization_id=user.selected_organization_id
    )
    if workflow is None:
        raise HTTPException(
            status_code=404, detail=f"Workflow with id {workflow_id} not found"
        )

    return {
        "id": workflow.id,
        "name": workflow.name,
        "status": workflow.status,
        "created_at": workflow.created_at,
        "workflow_definition": workflow.workflow_definition_with_fallback,
        "current_definition_id": workflow.current_definition_id,
        "template_context_variables": workflow.template_context_variables,
        "call_disposition_codes": workflow.call_disposition_codes,
        "workflow_configurations": workflow.workflow_configurations,
    }


@router.get("/summary")
async def get_workflows_summary(
    user: UserModel = Depends(get_user),
) -> List[WorkflowSummaryResponse]:
    """Get minimal workflow information (id and name only) for all workflows"""
    workflows = await db_client.get_all_workflows(
        organization_id=user.selected_organization_id
    )
    return [
        WorkflowSummaryResponse(id=workflow.id, name=workflow.name)
        for workflow in workflows
    ]


@router.put("/{workflow_id}/status")
async def update_workflow_status(
    workflow_id: int,
    request: UpdateWorkflowStatusRequest,
    user: UserModel = Depends(get_user),
) -> WorkflowResponse:
    """
    Update the status of a workflow (e.g., archive/unarchive).

    Args:
        workflow_id: The ID of the workflow to update
        request: The status update request

    Returns:
        The updated workflow
    """
    try:
        workflow = await db_client.update_workflow_status(
            workflow_id=workflow_id,
            status=request.status,
            organization_id=user.selected_organization_id,
        )
        run_count = await db_client.get_workflow_run_count(workflow.id)
        return {
            "id": workflow.id,
            "name": workflow.name,
            "status": workflow.status,
            "created_at": workflow.created_at,
            "workflow_definition": workflow.workflow_definition_with_fallback,
            "current_definition_id": workflow.current_definition_id,
            "template_context_variables": workflow.template_context_variables,
            "call_disposition_codes": workflow.call_disposition_codes,
            "workflow_configurations": workflow.workflow_configurations,
            "total_runs": run_count,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{workflow_id}")
async def update_workflow(
    workflow_id: int,
    request: UpdateWorkflowRequest,
    user: UserModel = Depends(get_user),
) -> WorkflowResponse:
    """
    Update an existing workflow.

    Args:
        workflow_id: The ID of the workflow to update
        request: The update request containing the new name and workflow definition

    Returns:
        The updated workflow

    Raises:
        HTTPException: If the workflow is not found or if there's a database error
    """
    try:
        workflow = await db_client.update_workflow(
            workflow_id=workflow_id,
            name=request.name,
            workflow_definition=request.workflow_definition,
            template_context_variables=request.template_context_variables,
            workflow_configurations=request.workflow_configurations,
            organization_id=user.selected_organization_id,
        )

        # Sync agent triggers if workflow definition was updated
        if request.workflow_definition:
            trigger_paths = extract_trigger_paths(request.workflow_definition)
            await db_client.sync_triggers_for_workflow(
                workflow_id=workflow.id,
                organization_id=user.selected_organization_id,
                trigger_paths=trigger_paths,
            )

        return {
            "id": workflow.id,
            "name": workflow.name,
            "status": workflow.status,
            "created_at": workflow.created_at,
            "workflow_definition": workflow.workflow_definition_with_fallback,
            "current_definition_id": workflow.current_definition_id,
            "template_context_variables": workflow.template_context_variables,
            "call_disposition_codes": workflow.call_disposition_codes,
            "workflow_configurations": workflow.workflow_configurations,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{workflow_id}/runs")
async def create_workflow_run(
    workflow_id: int,
    request: CreateWorkflowRunRequest,
    user: UserModel = Depends(get_user),
) -> CreateWorkflowRunResponse:
    """
    Create a new workflow run when the user decides to execute the workflow via chat or voice

    Args:
        workflow_id: The ID of the workflow to run
        request: The create workflow run request
        user: The user to create the workflow run for
    """
    run = await db_client.create_workflow_run(
        request.name, workflow_id, request.mode, user.id
    )
    return {
        "id": run.id,
        "workflow_id": run.workflow_id,
        "name": run.name,
        "mode": run.mode,
        "created_at": run.created_at,
        "definition_id": run.definition_id,
        "initial_context": run.initial_context,
        "gathered_context": run.gathered_context,
    }


@router.get("/{workflow_id}/runs/{run_id}")
async def get_workflow_run(
    workflow_id: int, run_id: int, user: UserModel = Depends(get_user)
) -> WorkflowRunResponseSchema:
    run = await db_client.get_workflow_run(
        run_id, organization_id=user.selected_organization_id
    )
    if not run:
        raise HTTPException(status_code=404, detail="Workflow run not found")
    return {
        "id": run.id,
        "workflow_id": run.workflow_id,
        "name": run.name,
        "mode": run.mode,
        "is_completed": run.is_completed,
        "transcript_url": run.transcript_url,
        "recording_url": run.recording_url,
        "cost_info": {
            "dograh_token_usage": (
                run.cost_info.get("dograh_token_usage")
                if run.cost_info and "dograh_token_usage" in run.cost_info
                else round(float(run.cost_info.get("total_cost_usd", 0)) * 100, 2)
                if run.cost_info and "total_cost_usd" in run.cost_info
                else 0
            ),
            "call_duration_seconds": int(
                round(run.cost_info.get("call_duration_seconds"))
            )
            if run.cost_info and run.cost_info.get("call_duration_seconds") is not None
            else None,
        }
        if run.cost_info
        else None,
        "created_at": run.created_at,
        "definition_id": run.definition_id,
        "initial_context": run.initial_context,
        "gathered_context": run.gathered_context,
        "call_type": run.call_type,
        "logs": run.logs,
    }


class WorkflowRunsResponse(BaseModel):
    runs: List[WorkflowRunResponseSchema]
    total_count: int
    page: int
    limit: int
    total_pages: int
    applied_filters: Optional[List[dict]] = None


@router.get("/{workflow_id}/runs")
async def get_workflow_runs(
    workflow_id: int,
    page: int = 1,
    limit: int = 50,
    filters: Optional[str] = Query(None, description="JSON-encoded filter criteria"),
    sort_by: Optional[str] = Query(
        None, description="Field to sort by (e.g., 'duration', 'created_at')"
    ),
    sort_order: Optional[str] = Query(
        "desc", description="Sort order ('asc' or 'desc')"
    ),
    user: UserModel = Depends(get_user),
) -> WorkflowRunsResponse:
    """
    Get workflow runs with optional filtering and sorting.

    Filters should be provided as a JSON-encoded array of filter criteria.
    Example: [{"attribute": "dateRange", "value": {"from": "2024-01-01", "to": "2024-01-31"}}]
    """
    offset = (page - 1) * limit

    # Parse filters if provided
    filter_criteria = []
    if filters:
        try:
            filter_criteria = json.loads(filters)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid filter format")

        # Restrict allowed filter attributes for regular users
        allowed_attributes = {
            "dateRange",
            "dispositionCode",
            "duration",
            "status",
            "tokenUsage",
        }
        for filter_item in filter_criteria:
            attribute = filter_item.get("attribute")
            if attribute and attribute not in allowed_attributes:
                raise HTTPException(
                    status_code=403, detail=f"Invalid attribute '{attribute}'"
                )

    runs, total_count = await db_client.get_workflow_runs_by_workflow_id(
        workflow_id,
        organization_id=user.selected_organization_id,
        limit=limit,
        offset=offset,
        filters=filter_criteria if filter_criteria else None,
        sort_by=sort_by,
        sort_order=sort_order,
    )

    total_pages = (total_count + limit - 1) // limit

    return WorkflowRunsResponse(
        runs=runs,
        total_count=total_count,
        page=page,
        limit=limit,
        total_pages=total_pages,
        applied_filters=filter_criteria if filter_criteria else None,
    )


@router.get("/templates")
async def get_workflow_templates() -> List[WorkflowTemplateResponse]:
    """
    Get all available workflow templates.

    Returns:
        List of workflow templates
    """
    template_client = WorkflowTemplateClient()
    templates = await template_client.get_all_workflow_templates()

    return [
        {
            "id": template.id,
            "template_name": template.template_name,
            "template_description": template.template_description,
            "template_json": template.template_json,
            "created_at": template.created_at,
        }
        for template in templates
    ]


@router.post("/templates/duplicate")
async def duplicate_workflow_template(
    request: DuplicateTemplateRequest, user: UserModel = Depends(get_user)
) -> WorkflowResponse:
    """
    Duplicate a workflow template to create a new workflow for the user.

    Args:
        request: The duplicate template request
        user: The authenticated user

    Returns:
        The newly created workflow
    """
    template_client = WorkflowTemplateClient()
    template = await template_client.get_workflow_template(request.template_id)

    if not template:
        raise HTTPException(
            status_code=404,
            detail=f"Workflow template with id {request.template_id} not found",
        )

    # Create a new workflow from the template
    # Regenerate trigger UUIDs to avoid conflicts with existing triggers
    workflow_def = regenerate_trigger_uuids(template.template_json)
    workflow = await db_client.create_workflow(
        request.workflow_name,
        workflow_def,
        user.id,
        user.selected_organization_id,
    )

    # Sync agent triggers if template contains any
    if workflow_def:
        trigger_paths = extract_trigger_paths(workflow_def)
        if trigger_paths:
            await db_client.sync_triggers_for_workflow(
                workflow_id=workflow.id,
                organization_id=user.selected_organization_id,
                trigger_paths=trigger_paths,
            )

    return {
        "id": workflow.id,
        "name": workflow.name,
        "status": workflow.status,
        "created_at": workflow.created_at,
        "workflow_definition": workflow.workflow_definition_with_fallback,
        "current_definition_id": workflow.current_definition_id,
        "template_context_variables": workflow.template_context_variables,
        "call_disposition_codes": workflow.call_disposition_codes,
        "workflow_configurations": workflow.workflow_configurations,
    }
