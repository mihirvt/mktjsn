import inspect
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from pydantic import BaseModel

from api.db import db_client
from api.db.models import (
    UserModel,
)
from api.services.auth.depends import get_user
from api.services.configuration.check_validity import (
    APIKeyStatusResponse,
    UserConfigurationValidator,
)
from api.services.configuration.defaults import DEFAULT_SERVICE_PROVIDERS
from api.services.configuration.masking import mask_user_config
from api.services.configuration.merge import merge_user_configurations
from api.services.configuration.registry import REGISTRY, ServiceType
from api.services.mps_service_key_client import mps_service_key_client

router = APIRouter(prefix="/user")


class AuthUserResponse(TypedDict):
    id: int
    is_superuser: bool


class DefaultConfigurationsResponse(TypedDict):
    llm: dict[str, dict]
    tts: dict[str, dict]
    stt: dict[str, dict]
    embeddings: dict[str, dict]
    default_providers: dict[str, str]


@router.get("/configurations/defaults")
async def get_default_configurations() -> DefaultConfigurationsResponse:
    configurations = {
        "llm": {
            provider: model_cls.model_json_schema()
            for provider, model_cls in REGISTRY[ServiceType.LLM].items()
        },
        "tts": {
            provider: model_cls.model_json_schema()
            for provider, model_cls in REGISTRY[ServiceType.TTS].items()
        },
        "stt": {
            provider: model_cls.model_json_schema()
            for provider, model_cls in REGISTRY[ServiceType.STT].items()
        },
        "embeddings": {
            provider: model_cls.model_json_schema()
            for provider, model_cls in REGISTRY[ServiceType.EMBEDDINGS].items()
        },
        "default_providers": DEFAULT_SERVICE_PROVIDERS,
    }
    return configurations


@router.get("/auth/user")
async def get_auth_user(
    user: UserModel = Depends(get_user),
) -> AuthUserResponse:
    return {
        "id": user.id,
        "is_superuser": user.is_superuser,
    }


class UserConfigurationRequestResponseSchema(BaseModel):
    llm: dict[str, Any] | None = None
    tts: dict[str, Any] | None = None
    stt: dict[str, Any] | None = None
    embeddings: dict[str, Any] | None = None
    test_phone_number: str | None = None
    timezone: str | None = None
    organization_pricing: dict[str, Union[float, str, bool]] | None = None


@router.get("/configurations/user")
async def get_user_configurations(
    user: UserModel = Depends(get_user),
) -> UserConfigurationRequestResponseSchema:
    user_configurations = await db_client.get_user_configurations(user.id)
    
    # Auto-provision if user signed up while MPS configuration generation was crashed
    # IMPORTANT: Use has_user_configuration rather than checking for empty fields 
    # to avoid overwriting existing configs that failed Pydantic validation!
    has_config = await db_client.has_user_configuration(user.id)
    if not has_config:
        from api.services.auth.depends import create_user_configuration_with_mps_key
        try:
            mps_config = await create_user_configuration_with_mps_key(
                user.id, user.selected_organization_id, user.provider_id
            )
            if mps_config:
                user_configurations = await db_client.update_user_configuration(user.id, mps_config)
        except Exception as e:
            logger.warning(f"Failed to auto-provision config on fetch: {e}")

    masked_config = mask_user_config(user_configurations)

    # Add organization pricing info if available
    if user.selected_organization_id:
        org = await db_client.get_organization_by_id(user.selected_organization_id)
        if org and org.price_per_second_usd is not None:
            masked_config["organization_pricing"] = {
                "price_per_second_usd": org.price_per_second_usd,
                "currency": "USD",
                "billing_enabled": True,
            }

    return masked_config


@router.put("/configurations/user")
async def update_user_configurations(
    request: UserConfigurationRequestResponseSchema,
    user: UserModel = Depends(get_user),
) -> UserConfigurationRequestResponseSchema:
    existing_config = await db_client.get_user_configurations(user.id)

    # Auto-provision if user signed up while MPS configuration generation was crashed
    # IMPORTANT: Use has_user_configuration rather than checking for empty fields 
    # to avoid overwriting existing configs that failed Pydantic validation!
    has_config = await db_client.has_user_configuration(user.id)
    if not has_config:
        from api.services.auth.depends import create_user_configuration_with_mps_key
        try:
            mps_config = await create_user_configuration_with_mps_key(
                user.id, user.selected_organization_id, user.provider_id
            )
            if mps_config:
                existing_config = await db_client.update_user_configuration(user.id, mps_config)
        except Exception as e:
            logger.warning(f"Failed to auto-provision config on put: {e}")

    incoming_dict = request.model_dump(exclude_none=True)

    # Remove organization_pricing from incoming dict as it's read-only
    incoming_dict.pop("organization_pricing", None)

    # Merge via helper
    user_configurations = merge_user_configurations(existing_config, incoming_dict)

    try:
        validator = UserConfigurationValidator()
        await validator.validate(user_configurations)
    except ValueError as e:
        logger.warning(
            f"User configuration validation failed for user_id={user.id}: {e.args[0]}"
        )
        raise HTTPException(status_code=422, detail=e.args[0])

    user_configurations = await db_client.update_user_configuration(
        user.id, user_configurations
    )

    # Return masked version of updated config
    masked_config = mask_user_config(user_configurations)

    # Add organization pricing info if available
    if user.selected_organization_id:
        org = await db_client.get_organization_by_id(user.selected_organization_id)
        if org and org.price_per_second_usd is not None:
            masked_config["organization_pricing"] = {
                "price_per_second_usd": org.price_per_second_usd,
                "currency": "USD",
                "billing_enabled": True,
            }

    return masked_config


@router.get("/configurations/user/validate")
async def validate_user_configurations(
    validity_ttl_seconds: int = Query(default=60, ge=0, le=86400),
    user: UserModel = Depends(get_user),
) -> APIKeyStatusResponse:
    configurations = await db_client.get_user_configurations(user.id)

    if (
        configurations.last_validated_at
        and configurations.last_validated_at
        < datetime.now() - timedelta(seconds=validity_ttl_seconds)
    ):
        validator = UserConfigurationValidator()
        try:
            status = await validator.validate(configurations)
            await db_client.update_user_configuration_last_validated_at(user.id)
            return status
        except ValueError as e:
            logger.warning(
                f"User configuration validation failed (validate endpoint) for user_id={user.id}: {e.args[0]}"
            )
            raise HTTPException(status_code=422, detail=e.args[0])
    else:
        return {"status": []}


# API Key Management Endpoints
class APIKeyResponse(BaseModel):
    id: int
    name: str
    key_prefix: str
    is_active: bool
    created_at: datetime
    last_used_at: Optional[datetime] = None
    archived_at: Optional[datetime] = None


class CreateAPIKeyRequest(BaseModel):
    name: str


class CreateAPIKeyResponse(BaseModel):
    id: int
    name: str
    key_prefix: str
    api_key: str  # Only returned when creating a new key
    created_at: datetime


@router.get("/api-keys")
async def get_api_keys(
    include_archived: bool = Query(default=False),
    user: UserModel = Depends(get_user),
) -> List[APIKeyResponse]:
    """Get all API keys for the user's selected organization."""
    if not user.selected_organization_id:
        raise HTTPException(status_code=400, detail="No organization selected")

    api_keys = await db_client.get_api_keys_by_organization(
        user.selected_organization_id, include_archived=include_archived
    )

    return [
        APIKeyResponse(
            id=key.id,
            name=key.name,
            key_prefix=key.key_prefix,
            is_active=key.is_active,
            created_at=key.created_at,
            last_used_at=key.last_used_at,
            archived_at=key.archived_at,
        )
        for key in api_keys
    ]


@router.post("/api-keys")
async def create_api_key(
    request: CreateAPIKeyRequest,
    user: UserModel = Depends(get_user),
) -> CreateAPIKeyResponse:
    """Create a new API key for the user's selected organization."""
    if not user.selected_organization_id:
        raise HTTPException(status_code=400, detail="No organization selected")

    api_key, raw_key = await db_client.create_api_key(
        organization_id=user.selected_organization_id,
        name=request.name,
        created_by=user.id,
    )

    return CreateAPIKeyResponse(
        id=api_key.id,
        name=api_key.name,
        key_prefix=api_key.key_prefix,
        api_key=raw_key,
        created_at=api_key.created_at,
    )


@router.delete("/api-keys/{api_key_id}")
async def archive_api_key(
    api_key_id: int,
    user: UserModel = Depends(get_user),
) -> dict:
    """Archive an API key (soft delete)."""
    if not user.selected_organization_id:
        raise HTTPException(status_code=400, detail="No organization selected")

    # Verify the API key belongs to the user's organization
    api_keys = await db_client.get_api_keys_by_organization(
        user.selected_organization_id, include_archived=True
    )
    if not any(key.id == api_key_id for key in api_keys):
        raise HTTPException(status_code=404, detail="API key not found")

    success = await db_client.archive_api_key(api_key_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to archive API key")

    return {"success": True, "message": "API key archived successfully"}


@router.put("/api-keys/{api_key_id}/reactivate")
async def reactivate_api_key(
    api_key_id: int,
    user: UserModel = Depends(get_user),
) -> dict:
    """Reactivate an archived API key."""
    if not user.selected_organization_id:
        raise HTTPException(status_code=400, detail="No organization selected")

    # Verify the API key belongs to the user's organization
    api_keys = await db_client.get_api_keys_by_organization(
        user.selected_organization_id, include_archived=True
    )
    if not any(key.id == api_key_id for key in api_keys):
        raise HTTPException(status_code=404, detail="API key not found")

    success = await db_client.reactivate_api_key(api_key_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to reactivate API key")

    return {"success": True, "message": "API key reactivated successfully"}


# Voice Configuration Endpoints
TTSProvider = Literal[
    "elevenlabs",
    "deepgram",
    "fish",
    "sarvam",
    "cartesia",
    "dograh",
    "smallest_ai",
    "voicemaker",
    "murf",
    "grok",
    "inworld",
]


class VoiceInfo(BaseModel):
    voice_id: str
    name: str
    description: Optional[str] = None
    accent: Optional[str] = None
    gender: Optional[str] = None
    language: Optional[str] = None
    preview_url: Optional[str] = None
    supported_locales: Optional[List[str]] = None
    styles_by_locale: Optional[Dict[str, List[str]]] = None


class VoicesResponse(BaseModel):
    provider: str
    voices: List[VoiceInfo]


@router.get("/configurations/voices/{provider}")
async def get_voices(
    provider: TTSProvider,
    user: UserModel = Depends(get_user),
) -> VoicesResponse:
    """Get available voices for a TTS provider."""
    try:
        if provider == "smallest_ai":
            user_config = await db_client.get_user_configurations(user.id)
            api_key = None
            if user_config.tts:
                provider_val = user_config.tts.provider.value if hasattr(user_config.tts.provider, "value") else user_config.tts.provider
                if provider_val == "smallest_ai":
                    api_key = getattr(user_config.tts, "api_key", None)
            
            if not api_key:
                 return VoicesResponse(provider="smallest_ai", voices=[])
                 
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.get(
                    "https://waves-api.smallest.ai/api/v1/lightning-v3.1/get_voices",
                    headers={"Authorization": f"Bearer {api_key}"}
                )
                if res.status_code == 200:
                    data = res.json()
                    voice_list = data if isinstance(data, list) else data.get("voices", [])
                    voices = []
                    for v in voice_list:
                        tags = v.get("tags", {})
                        languages = tags.get("language", [])
                        voices.append(VoiceInfo(
                            voice_id=v.get("voiceId", v.get("voice_id", "unknown")),
                            name=v.get("displayName", v.get("name", "Unknown Voice")),
                            gender=tags.get("gender"),
                            accent=tags.get("accent"),
                            language=", ".join(languages) if isinstance(languages, list) else languages,
                        ))
                    return VoicesResponse(provider="smallest_ai", voices=voices)
                else:
                    logger.error(f"Failed to fetch Smallest.ai voices: {res.status_code} {res.text}")
                    return VoicesResponse(provider="smallest_ai", voices=[])

        if provider == "fish":
            user_config = await db_client.get_user_configurations(user.id)
            api_key = None
            if user_config.tts:
                provider_val = (
                    user_config.tts.provider.value
                    if hasattr(user_config.tts.provider, "value")
                    else user_config.tts.provider
                )
                if provider_val == "fish":
                    api_key = getattr(user_config.tts, "api_key", None)

            if not api_key:
                return VoicesResponse(provider="fish", voices=[])

            try:
                from fishaudio import AsyncFishAudio
            except ImportError:
                logger.error(
                    "Fish Audio SDK not installed. Please add fish-audio-sdk to use voice discovery."
                )
                return VoicesResponse(provider="fish", voices=[])

            client = AsyncFishAudio(api_key=api_key)
            try:
                response = await client.voices.list()
                voices = []
                for voice in response.items:
                    if getattr(voice, "type", None) != "tts":
                        continue

                    preview_url = None
                    samples = getattr(voice, "samples", None) or []
                    if samples:
                        preview_url = getattr(samples[0], "audio", None)

                    tags = list(getattr(voice, "tags", None) or [])
                    gender = next(
                        (tag for tag in tags if tag in {"male", "female", "neutral"}),
                        None,
                    )

                    voices.append(
                        VoiceInfo(
                            voice_id=voice.id,
                            name=getattr(voice, "title", voice.id),
                            description=getattr(voice, "description", None),
                            gender=gender,
                            language=", ".join(getattr(voice, "languages", []) or []),
                            preview_url=preview_url,
                        )
                    )
                return VoicesResponse(provider="fish", voices=voices)
            finally:
                close_method = (
                    getattr(client, "aclose", None) or getattr(client, "close", None)
                )
                if close_method:
                    close_result = close_method()
                    if inspect.isawaitable(close_result):
                        await close_result

        if provider == "voicemaker":
            user_config = await db_client.get_user_configurations(user.id)
            api_key = None
            if user_config.tts:
                provider_val = (
                    user_config.tts.provider.value
                    if hasattr(user_config.tts.provider, "value")
                    else user_config.tts.provider
                )
                if provider_val == "voicemaker":
                    api_key = getattr(user_config.tts, "api_key", None)

            if not api_key:
                return VoicesResponse(provider="voicemaker", voices=[])

            import httpx

            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.post(
                    "https://developer.voicemaker.in/api/v1/voice/list",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={},  # no filter = all voices
                )

            if res.status_code == 200:
                data = res.json()
                voice_list = data.get("data", {}).get("voices_list", [])
                logger.debug(
                    f"Voicemaker: fetched {len(voice_list)} voices"
                )
                voices = []
                for v in voice_list:
                    voices.append(
                        VoiceInfo(
                            voice_id=v.get("VoiceId", ""),
                            name=v.get("VoiceWebname", v.get("VoiceId", "")),
                            gender=v.get("VoiceGender"),
                            language=v.get("Language"),
                            accent=v.get("Country"),
                            description=v.get("Engine"),
                        )
                    )
                return VoicesResponse(provider="voicemaker", voices=voices)
            else:
                logger.error(
                    f"Voicemaker voice list API error: {res.status_code} {res.text[:300]}"
                )
                return VoicesResponse(provider="voicemaker", voices=[])

        if provider == "murf":
            user_config = await db_client.get_user_configurations(user.id)
            api_key = None
            if user_config.tts:
                provider_val = (
                    user_config.tts.provider.value
                    if hasattr(user_config.tts.provider, "value")
                    else user_config.tts.provider
                )
                if provider_val == "murf":
                    api_key = getattr(user_config.tts, "api_key", None)

            if not api_key:
                return VoicesResponse(provider="murf", voices=[])

            import httpx

            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.get(
                    "https://api.murf.ai/v1/speech/voices",
                    params={"model": "FALCON"},
                    headers={"api-key": api_key},
                )

            if res.status_code == 200:
                voice_list = res.json()
                voices = []
                for voice in voice_list:
                    supported_locales_raw = voice.get("supportedLocales") or {}
                    supported_locales = list(supported_locales_raw.keys())
                    styles_by_locale = {
                        locale: locale_details.get("availableStyles", [])
                        for locale, locale_details in supported_locales_raw.items()
                    }
                    primary_locale = voice.get("locale") or (supported_locales[0] if supported_locales else None)
                    voices.append(
                        VoiceInfo(
                            voice_id=voice.get("voiceId", ""),
                            name=voice.get("displayName", voice.get("voiceId", "")),
                            description=voice.get("description"),
                            accent=voice.get("accent"),
                            gender=voice.get("gender"),
                            language=primary_locale,
                            supported_locales=supported_locales or None,
                            styles_by_locale=styles_by_locale or None,
                        )
                    )
                return VoicesResponse(provider="murf", voices=voices)

            logger.error(f"Failed to fetch Murf voices: {res.status_code} {res.text[:300]}")
            return VoicesResponse(provider="murf", voices=[])

        if provider == "grok":
            # xAI Grok has 5 static voices — no API call needed
            grok_voices = [
                VoiceInfo(
                    voice_id="eve",
                    name="Eve",
                    description="Warm, conversational female voice",
                    gender="female",
                ),
                VoiceInfo(
                    voice_id="ara",
                    name="Ara",
                    description="Clear, professional female voice",
                    gender="female",
                ),
                VoiceInfo(
                    voice_id="rex",
                    name="Rex",
                    description="Deep, authoritative male voice",
                    gender="male",
                ),
                VoiceInfo(
                    voice_id="sal",
                    name="Sal",
                    description="Friendly, approachable male voice",
                    gender="male",
                ),
                VoiceInfo(
                    voice_id="leo",
                    name="Leo",
                    description="Energetic, expressive male voice",
                    gender="male",
                ),
            ]
            return VoicesResponse(provider="grok", voices=grok_voices)

        if provider == "inworld":
            user_config = await db_client.get_user_configurations(user.id)
            api_key = None
            if user_config.tts:
                provider_val = (
                    user_config.tts.provider.value
                    if hasattr(user_config.tts.provider, "value")
                    else user_config.tts.provider
                )
                if provider_val == "inworld":
                    api_key = getattr(user_config.tts, "api_key", None)

            if not api_key:
                return VoicesResponse(provider="inworld", voices=[])

            import httpx

            # Fetch all voices — no language filter so custom cloned voices are included
            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.get(
                    "https://api.inworld.ai/tts/v1/voices",
                    headers={"Authorization": f"Basic {api_key}"},
                )

            if res.status_code == 200:
                data = res.json()
                voice_list = data.get("voices", [])
                voices = []
                for v in voice_list:
                    tags = v.get("tags", [])
                    gender = next(
                        (tag for tag in tags if tag in {"male", "female", "neutral"}),
                        None,
                    )
                    languages = v.get("languages", [])
                    voices.append(
                        VoiceInfo(
                            voice_id=v.get("voiceId", ""),
                            name=v.get("displayName", v.get("voiceId", "")),
                            description=v.get("description"),
                            gender=gender,
                            language=", ".join(languages) if languages else None,
                        )
                    )
                return VoicesResponse(provider="inworld", voices=voices)
            else:
                logger.error(
                    f"Failed to fetch Inworld voices: {res.status_code} {res.text[:300]}"
                )
                return VoicesResponse(provider="inworld", voices=[])

        result = await mps_service_key_client.get_voices(
            provider=provider,
            organization_id=user.selected_organization_id,
            created_by=user.provider_id,
        )
        return VoicesResponse(
            provider=result.get("provider", provider),
            voices=[VoiceInfo(**voice) for voice in result.get("voices", [])],
        )
    except Exception as e:
        logger.error(f"Failed to fetch voices for {provider}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch voices for {provider}",
        )
