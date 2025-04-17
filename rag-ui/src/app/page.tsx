'use client';

import React, { useState, useEffect } from 'react';
import type { QueryResponse } from '@/types';
import { ChatInput } from '@/components/ChatInput';
import { ChatMessages } from '@/components/ChatMessages';

// API base URL from environment variable or default
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Home() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [responses, setResponses] = useState<QueryResponse[]>([]);

  const handleSubmit = async (query: string, context?: string) => {
    setLoading(true);
    setError(null);

    try {
      // Start workflow
      const startResponse = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          context: context || null,
        }),
      });

      if (!startResponse.ok) {
        throw new Error(`Failed to start workflow: ${startResponse.statusText}`);
      }

      const { workflow_id } = await startResponse.json();
      let retryCount = 0;
      const maxRetries = 5; // Number of retries before starting to poll
      
      // Initial retries for quick response
      while (retryCount < maxRetries) {
        const resultResponse = await fetch(`${API_BASE_URL}/query/${workflow_id}`);
        if (!resultResponse.ok) {
          throw new Error(`Failed to get results: ${resultResponse.statusText}`);
        }
        
        const result = await resultResponse.json();
        
        if (result.status === 'completed') {
          setResponses(prev => [...prev, {
            query,
            context,
            response: result.result,
            timestamp: new Date().toISOString(),
            workflow_id
          }]);
          setLoading(false);
          return;
        } else if (result.status === 'failed' || result.status === 'error') {
          setError(`Query processing failed: ${result.message || result.error || 'Unknown error'}`);
          setLoading(false);
          return;
        } else if (result.status === 'not_found') {
          setError(`Workflow not found: ${result.message || 'Unknown error'}`);
          setLoading(false);
          return;
        }
        
        retryCount++;
        await new Promise(resolve => setTimeout(resolve, 200)); // 200ms between quick retries
      }

      // If not completed after quick retries, start polling
      const pollInterval = setInterval(async () => {
        try {
          const resultResponse = await fetch(`${API_BASE_URL}/query/${workflow_id}`);
          
          if (!resultResponse.ok) {
            throw new Error(`Failed to get results: ${resultResponse.statusText}`);
          }

          const result = await resultResponse.json();

          if (result.status === 'completed') {
            clearInterval(pollInterval);
            setResponses(prev => [...prev, {
              query,
              context,
              response: result.result,
              timestamp: new Date().toISOString(),
              workflow_id
            }]);
            setLoading(false);
          } else if (result.status === 'failed' || result.status === 'error') {
            clearInterval(pollInterval);
            setError(`Query processing failed: ${result.message || result.error || 'Unknown error'}`);
            setLoading(false);
          } else if (result.status === 'not_found') {
            clearInterval(pollInterval);
            setError(`Workflow not found: ${result.message || 'Unknown error'}`);
            setLoading(false);
          } else if (result.status === 'initializing') {
            console.log('Workflow is initializing...');
          }
        } catch (error) {
          console.error('Polling error:', error);
          // Don't clear interval or set error on transient errors
        }
      }, 1000);

      // Cleanup interval after 30 seconds to prevent infinite polling
      setTimeout(() => {
        clearInterval(pollInterval);
        if (loading) {
          setError('Request timed out after 30 seconds');
          setLoading(false);
        }
      }, 30000);

    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to process query');
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gray-100">
      <div className="max-w-4xl mx-auto p-4">
        <h1 className="text-3xl font-bold text-center mb-8 text-gray-800">
          Chevy Colorado 2022 Assistant
        </h1>

        <div className="bg-white rounded-lg shadow-lg overflow-hidden">
          {error && (
            <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-4">
              <p className="text-red-700">{error}</p>
            </div>
          )}
          
          <ChatMessages 
            responses={responses} 
            loading={loading} 
          />
          <ChatInput 
            onSubmit={handleSubmit}
            disabled={loading}
          />
        </div>
      </div>
    </main>
  );
} 