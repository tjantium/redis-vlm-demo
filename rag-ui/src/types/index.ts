export interface QueryResponse {
    query: string;
    context?: string;
    response: string;
    timestamp: string;
    workflow_id: string;
}

export interface WorkflowResponse {
    workflow_id: string;
    status: string;
    result: string;
    metrics?: {
        duration_seconds: number;
        response_length: number;
        event_count: number;
    };
    events?: Array<{
        timestamp: string;
        event_type: string;
        details: Record<string, any>;
    }>;
} 