import React, { useEffect, useRef } from 'react';
import type { QueryResponse } from '../types';

interface ChatMessagesProps {
  responses: QueryResponse[];
  loading: boolean;
}

export function ChatMessages({ responses, loading }: ChatMessagesProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [responses, loading]);

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  return (
    <div className="min-h-[400px] max-h-[600px] overflow-y-auto p-4 bg-gray-50">
      {responses.map((resp, index) => (
        <div 
          key={`${resp.workflow_id}-${index}`} 
          className="mb-6 animate-fade-in"
        >
          <div className="flex flex-col space-y-3">
            {/* User Message */}
            <div className="flex flex-col items-end space-y-1">
              <div className="bg-blue-500 text-white p-3 rounded-lg rounded-tr-none max-w-[85%] shadow-sm">
                <p className="text-sm">{resp.query}</p>
                {resp.context && (
                  <div className="mt-2 pt-2 border-t border-blue-400">
                    <p className="text-xs opacity-80">Context: {resp.context}</p>
                  </div>
                )}
              </div>
              <span className="text-xs text-gray-500">
                {formatTimestamp(resp.timestamp)}
              </span>
            </div>

            {/* Assistant Response */}
            <div className="flex flex-col items-start space-y-1">
              <div className="bg-white p-3 rounded-lg rounded-tl-none max-w-[85%] shadow-sm">
                <p className="whitespace-pre-wrap text-gray-800">{resp.response}</p>
              </div>
              <span className="text-xs text-gray-500">
                Workflow ID: {resp.workflow_id}
              </span>
            </div>
          </div>
        </div>
      ))}
      
      {loading && (
        <div className="flex justify-center items-center py-4">
          <div className="flex items-center space-x-2">
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-500"></div>
            <span className="text-sm text-gray-500">Processing your question...</span>
          </div>
        </div>
      )}
      
      {!loading && responses.length === 0 && (
        <div className="flex flex-col items-center justify-center h-full text-gray-500 space-y-4">
          <svg 
            className="h-12 w-12 text-gray-400" 
            fill="none" 
            viewBox="0 0 24 24" 
            stroke="currentColor"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={2} 
              d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" 
            />
          </svg>
          <p className="text-center">
            Ask a question about the Chevy Colorado 2022
            <br />
            <span className="text-sm">Try asking about features, specifications, or maintenance</span>
          </p>
        </div>
      )}

      <div ref={messagesEndRef} />
    </div>
  );
} 