import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { AmbientNoiseConfiguration, TurnStopStrategy, VADConfiguration, WorkflowConfigurations } from "@/types/workflow-configurations";

interface ConfigurationsDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    workflowConfigurations: WorkflowConfigurations | null;
    workflowName: string;
    onSave: (configurations: WorkflowConfigurations, workflowName: string) => Promise<void>;
}

const DEFAULT_AMBIENT_NOISE_CONFIG: AmbientNoiseConfiguration = {
    enabled: false,
    volume: 0.3,
};

const DEFAULT_VAD_CONFIG: VADConfiguration = {
    confidence: 0.7,
    start_secs: 0.2,
    stop_secs: 0.2,
    min_volume: 0.2,
};

export const ConfigurationsDialog = ({
    open,
    onOpenChange,
    workflowConfigurations,
    workflowName,
    onSave
}: ConfigurationsDialogProps) => {
    const [name, setName] = useState<string>(workflowName);
    const [ambientNoiseConfig, setAmbientNoiseConfig] = useState<AmbientNoiseConfiguration>(
        workflowConfigurations?.ambient_noise_configuration || DEFAULT_AMBIENT_NOISE_CONFIG
    );
    const [vadConfig, setVadConfig] = useState<VADConfiguration>(
        workflowConfigurations?.vad_configuration || DEFAULT_VAD_CONFIG
    );
    const [maxCallDuration, setMaxCallDuration] = useState<number>(
        workflowConfigurations?.max_call_duration || 600  // Default 10 minutes
    );
    const [maxUserIdleTimeout, setMaxUserIdleTimeout] = useState<number>(
        workflowConfigurations?.max_user_idle_timeout || 10  // Default 10 seconds
    );
    const [smartTurnStopSecs, setSmartTurnStopSecs] = useState<number>(
        workflowConfigurations?.smart_turn_stop_secs || 2  // Default 2 seconds
    );
    const [turnStopStrategy, setTurnStopStrategy] = useState<TurnStopStrategy>(
        workflowConfigurations?.turn_stop_strategy || 'transcription'
    );
    const [isSaving, setIsSaving] = useState(false);

    const handleSave = async () => {
        setIsSaving(true);
        try {
            await onSave({
                ambient_noise_configuration: ambientNoiseConfig,
                vad_configuration: vadConfig,
                max_call_duration: maxCallDuration,
                max_user_idle_timeout: maxUserIdleTimeout,
                smart_turn_stop_secs: smartTurnStopSecs,
                turn_stop_strategy: turnStopStrategy
            }, name);
            onOpenChange(false);
        } catch (error) {
            console.error("Failed to save configurations:", error);
        } finally {
            setIsSaving(false);
        }
    };

    // Sync state with props when dialog opens
    useEffect(() => {
        if (open) {
            setName(workflowName);
            setAmbientNoiseConfig(workflowConfigurations?.ambient_noise_configuration || DEFAULT_AMBIENT_NOISE_CONFIG);
            setVadConfig(workflowConfigurations?.vad_configuration || DEFAULT_VAD_CONFIG);
            setMaxCallDuration(workflowConfigurations?.max_call_duration || 600);
            setMaxUserIdleTimeout(workflowConfigurations?.max_user_idle_timeout || 10);
            setSmartTurnStopSecs(workflowConfigurations?.smart_turn_stop_secs || 2);
            setTurnStopStrategy(workflowConfigurations?.turn_stop_strategy || 'transcription');
        }
    }, [open, workflowName, workflowConfigurations]);

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="max-w-lg">
                <DialogHeader>
                    <DialogTitle>Configurations</DialogTitle>
                </DialogHeader>

                <div className="space-y-6">
                    {/* Workflow Name Section */}
                    <div className="space-y-4">
                        <div>
                            <h3 className="text-sm font-semibold mb-1">Agent Name</h3>
                            <p className="text-xs text-muted-foreground">
                                The name of your agent
                            </p>
                        </div>
                        <div className="space-y-2">
                            <Label htmlFor="workflow_name" className="text-xs">
                                Name
                            </Label>
                            <Input
                                id="workflow_name"
                                type="text"
                                value={name}
                                onChange={(e) => setName(e.target.value)}
                                placeholder="Enter Agent name"
                            />
                        </div>
                    </div>

                    {/* Ambient Noise Section */}
                    <div className="space-y-4">
                        <div>
                            <h3 className="text-sm font-semibold mb-1">Ambient Noise</h3>
                            <p className="text-xs text-muted-foreground">
                                Add background office ambient noise to make the conversation sound more natural.
                            </p>
                        </div>

                        <div className="space-y-4">
                            <div className="flex items-center justify-between">
                                <Label htmlFor="ambient-noise-enabled" className="text-sm">
                                    Use Ambient Noise
                                </Label>
                                <Switch
                                    id="ambient-noise-enabled"
                                    checked={ambientNoiseConfig.enabled}
                                    onCheckedChange={(checked) =>
                                        setAmbientNoiseConfig(prev => ({ ...prev, enabled: checked }))
                                    }
                                />
                            </div>

                            {ambientNoiseConfig.enabled && (
                                <div className="space-y-2">
                                    <Label htmlFor="ambient-volume" className="text-xs">
                                        Volume
                                    </Label>
                                    <Input
                                        id="ambient-volume"
                                        type="number"
                                        step="0.1"
                                        min="0"
                                        max="1"
                                        value={ambientNoiseConfig.volume}
                                        onChange={(e) => {
                                            const value = parseFloat(e.target.value);
                                            if (!isNaN(value)) {
                                                setAmbientNoiseConfig(prev => ({ ...prev, volume: value }));
                                            }
                                        }}
                                    />
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Turn Detection Section */}
                    <div className="space-y-4">
                        <div>
                            <h3 className="text-sm font-semibold mb-1">Turn Detection (VAD)</h3>
                            <p className="text-xs text-muted-foreground">
                                Configure how the agent detects when the user has finished speaking and voice activity parameters.
                            </p>
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                            <div className="space-y-2">
                                <Label htmlFor="vad_min_volume" className="text-xs">
                                    VAD Minimum Volume (0-1)
                                </Label>
                                <Input
                                    id="vad_min_volume"
                                    type="number"
                                    step="0.01"
                                    min="0"
                                    max="1"
                                    value={vadConfig.min_volume}
                                    onChange={(e) => {
                                        const value = parseFloat(e.target.value);
                                        if (!isNaN(value)) setVadConfig(prev => ({ ...prev, min_volume: value }));
                                    }}
                                />
                                <p className="text-xs text-muted-foreground">Volume threshold for speech (Def: 0.2 for Phone)</p>
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="vad_confidence" className="text-xs">
                                    VAD Confidence (0-1)
                                </Label>
                                <Input
                                    id="vad_confidence"
                                    type="number"
                                    step="0.01"
                                    min="0"
                                    max="1"
                                    value={vadConfig.confidence}
                                    onChange={(e) => {
                                        const value = parseFloat(e.target.value);
                                        if (!isNaN(value)) setVadConfig(prev => ({ ...prev, confidence: value }));
                                    }}
                                />
                                <p className="text-xs text-muted-foreground">Speech detection confidence (Def: 0.7)</p>
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="vad_start_secs" className="text-xs">
                                    VAD Start Seconds
                                </Label>
                                <Input
                                    id="vad_start_secs"
                                    type="number"
                                    step="0.1"
                                    min="0"
                                    value={vadConfig.start_secs}
                                    onChange={(e) => {
                                        const value = parseFloat(e.target.value);
                                        if (!isNaN(value)) setVadConfig(prev => ({ ...prev, start_secs: value }));
                                    }}
                                />
                                <p className="text-xs text-muted-foreground">Seconds of speech to start turn (Def: 0.2)</p>
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="vad_stop_secs" className="text-xs">
                                    VAD Stop Seconds
                                </Label>
                                <Input
                                    id="vad_stop_secs"
                                    type="number"
                                    step="0.1"
                                    min="0"
                                    value={vadConfig.stop_secs}
                                    onChange={(e) => {
                                        const value = parseFloat(e.target.value);
                                        if (!isNaN(value)) setVadConfig(prev => ({ ...prev, stop_secs: value }));
                                    }}
                                />
                                <p className="text-xs text-muted-foreground">Seconds of silence to end turn (Def: 0.2)</p>
                            </div>
                        </div>

                        <div className="space-y-2 pt-2">
                            <Label htmlFor="turn_stop_strategy" className="text-xs">
                                Turn Disruption Strategy
                            </Label>
                            <Select
                                value={turnStopStrategy}
                                onValueChange={(value: TurnStopStrategy) => setTurnStopStrategy(value)}
                            >
                                <SelectTrigger id="turn_stop_strategy">
                                    <SelectValue placeholder="Select strategy" />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="transcription">
                                        Transcription-based
                                    </SelectItem>
                                    <SelectItem value="turn_analyzer">
                                        Smart Turn Analyzer
                                    </SelectItem>
                                </SelectContent>
                            </Select>
                            <p className="text-xs text-muted-foreground">
                                {turnStopStrategy === 'transcription'
                                    ? "Best for short responses (1-2 word statements). Ends turn when transcription indicates completion."
                                    : "Best for longer responses with natural pauses. Uses ML model to detect end of turn."}
                            </p>
                        </div>

                        {turnStopStrategy === 'turn_analyzer' && (
                            <div className="space-y-2">
                                <Label htmlFor="smart_turn_stop_secs" className="text-xs">
                                    Incomplete Turn Timeout (seconds)
                                </Label>
                                <Input
                                    id="smart_turn_stop_secs"
                                    type="number"
                                    step="0.5"
                                    min="0.5"
                                    max="10"
                                    value={smartTurnStopSecs}
                                    onChange={(e) => {
                                        const value = parseFloat(e.target.value);
                                        if (!isNaN(value) && value >= 0.5) {
                                            setSmartTurnStopSecs(value);
                                        }
                                    }}
                                />
                                <p className="text-xs text-muted-foreground">
                                    Max silence duration before ending an incomplete turn. Default: 2 seconds
                                </p>
                            </div>
                        )}
                    </div>

                    {/* Call Management Section */}
                    <div className="space-y-4">
                        <div>
                            <h3 className="text-sm font-semibold mb-1">Call Management</h3>
                            <p className="text-xs text-muted-foreground">
                                Configure call duration limits and idle timeout settings.
                            </p>
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                            <div className="space-y-2">
                                <Label htmlFor="max_call_duration" className="text-xs">
                                    Max Call Duration (seconds)
                                </Label>
                                <Input
                                    id="max_call_duration"
                                    type="number"
                                    step="1"
                                    min="1"
                                    value={maxCallDuration}
                                    onChange={(e) => {
                                        const value = parseInt(e.target.value);
                                        if (!isNaN(value) && value > 0) {
                                            setMaxCallDuration(value);
                                        }
                                    }}
                                />
                                <p className="text-xs text-muted-foreground">Default: 600 (10 minutes)</p>
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="max_user_idle_timeout" className="text-xs">
                                    Max User Idle Timeout (seconds)
                                </Label>
                                <Input
                                    id="max_user_idle_timeout"
                                    type="number"
                                    step="1"
                                    min="1"
                                    value={maxUserIdleTimeout}
                                    onChange={(e) => {
                                        const value = parseInt(e.target.value);
                                        if (!isNaN(value) && value > 0) {
                                            setMaxUserIdleTimeout(value);
                                        }
                                    }}
                                />
                                <p className="text-xs text-muted-foreground">Default: 10 seconds</p>
                            </div>
                        </div>
                    </div>
                </div>

                <DialogFooter>
                    <Button variant="outline" onClick={() => onOpenChange(false)}>
                        Cancel
                    </Button>
                    <Button onClick={handleSave} disabled={isSaving}>
                        {isSaving ? "Saving..." : "Save"}
                    </Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
};

