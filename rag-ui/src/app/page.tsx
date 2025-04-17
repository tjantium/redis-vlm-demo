'use client';

import React, { useState, useEffect } from 'react';
import type { QueryResponse } from '@/types';
import { ChatInput } from '@/components/ChatInput';
import { ChatMessages } from '@/components/ChatMessages';

// API base URL from environment variable or default
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Configuration
const CONFIG = {
  POLL_INTERVAL: 1000,      // Poll every second
  MAX_QUICK_RETRIES: 5,     // Number of quick retries
  QUICK_RETRY_DELAY: 200,   // 200ms between quick retries
  MAX_POLL_TIME: 90000,     // Maximum polling time (90 seconds)
  INITIAL_TIMEOUT: 90000,   // Initial timeout (90 seconds)
  PROGRESS_UPDATE_INTERVAL: 10000 // Update progress every 10 seconds
};

const extractAnswer = (result: string): string => {
  // If response contains "Answer: ", extract everything after it
  if (result.includes("Answer: ")) {
    return result.split("Answer: ")[1].trim();
  }
  // If response contains "Final Answer: ", extract everything after it
  if (result.includes("Final Answer: ")) {
    return result.split("Final Answer: ")[1].trim();
  }
  // Remove any "Thought:" sections
  if (result.includes("Thought:")) {
    const parts = result.split("Thought:");
    return parts[parts.length - 1].trim();
  }
  return result.trim();
};

export default function Home() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [responses, setResponses] = useState<QueryResponse[]>([]);

  const handleSubmit = async (query: string, context?: string) => {
    setLoading(true);
    setError(null);

    let pollInterval: NodeJS.Timeout | null = null;
    let timeoutId: NodeJS.Timeout | null = null;
    let progressUpdateId: NodeJS.Timeout | null = null;
    let startTime = Date.now();

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
      
      // Initial retries for quick response
      while (retryCount < CONFIG.MAX_QUICK_RETRIES) {
        const resultResponse = await fetch(`${API_BASE_URL}/query/${workflow_id}`);
        if (!resultResponse.ok) {
          throw new Error(`Failed to get results: ${resultResponse.statusText}`);
        }
        
        const result = await resultResponse.json();
        
        if (result.status === 'completed') {
          if (pollInterval) clearInterval(pollInterval);
          if (timeoutId) clearTimeout(timeoutId);
          if (progressUpdateId) clearInterval(progressUpdateId);
          
          const cleanResponse = extractAnswer(result.result);
          if (cleanResponse.startsWith("Error:")) {
            setError(cleanResponse);
            setLoading(false);
            return;
          }
          setResponses(prev => [...prev, {
            query,
            context,
            response: cleanResponse,
            timestamp: new Date().toISOString(),
            workflow_id
          }]);
          setLoading(false);
          return;
        } else if (result.status === 'failed' || result.status === 'error') {
          if (pollInterval) clearInterval(pollInterval);
          if (timeoutId) clearTimeout(timeoutId);
          if (progressUpdateId) clearInterval(progressUpdateId);
          setError(`Query processing failed: ${result.message || result.error || 'Unknown error'}`);
          setLoading(false);
          return;
        } else if (result.status === 'not_found') {
          if (pollInterval) clearInterval(pollInterval);
          if (timeoutId) clearTimeout(timeoutId);
          if (progressUpdateId) clearInterval(progressUpdateId);
          setError(`Workflow not found: ${result.message || 'Unknown error'}`);
          setLoading(false);
          return;
        }
        
        retryCount++;
        await new Promise(resolve => setTimeout(resolve, CONFIG.QUICK_RETRY_DELAY));
      }

      // Start progress updates
      progressUpdateId = setInterval(() => {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        console.log(`Query processing in progress... (${elapsed}s)`);
      }, CONFIG.PROGRESS_UPDATE_INTERVAL);

      // If not completed after quick retries, start polling
      pollInterval = setInterval(async () => {
        try {
          const resultResponse = await fetch(`${API_BASE_URL}/query/${workflow_id}`);
          
          if (!resultResponse.ok) {
            throw new Error(`Failed to get results: ${resultResponse.statusText}`);
          }

          const result = await resultResponse.json();

          if (result.status === 'completed') {
            if (pollInterval) clearInterval(pollInterval);
            if (timeoutId) clearTimeout(timeoutId);
            if (progressUpdateId) clearInterval(progressUpdateId);
            
            const cleanResponse = extractAnswer(result.result);
            if (cleanResponse.startsWith("Error:")) {
              setError(cleanResponse);
              setLoading(false);
              return;
            }
            setResponses(prev => [...prev, {
              query,
              context,
              response: cleanResponse,
              timestamp: new Date().toISOString(),
              workflow_id
            }]);
            setLoading(false);
          } else if (result.status === 'failed' || result.status === 'error') {
            if (pollInterval) clearInterval(pollInterval);
            if (timeoutId) clearTimeout(timeoutId);
            if (progressUpdateId) clearInterval(progressUpdateId);
            setError(`Query processing failed: ${result.message || result.error || 'Unknown error'}`);
            setLoading(false);
          } else if (result.status === 'not_found') {
            if (pollInterval) clearInterval(pollInterval);
            if (timeoutId) clearTimeout(timeoutId);
            if (progressUpdateId) clearInterval(progressUpdateId);
            setError(`Workflow not found: ${result.message || 'Unknown error'}`);
            setLoading(false);
          } else if (result.status === 'initializing') {
            console.log('Workflow is initializing...');
          }
        } catch (error) {
          console.error('Polling error:', error);
          // Don't clear interval or set error on transient errors
        }
      }, CONFIG.POLL_INTERVAL);

      // Set timeout for polling
      timeoutId = setTimeout(() => {
        if (pollInterval) clearInterval(pollInterval);
        if (progressUpdateId) clearInterval(progressUpdateId);
        if (loading) {
          setError('Request timed out. The response is taking longer than expected. Please try again.');
          setLoading(false);
        }
      }, CONFIG.MAX_POLL_TIME);

    } catch (error) {
      if (pollInterval) clearInterval(pollInterval);
      if (timeoutId) clearTimeout(timeoutId);
      if (progressUpdateId) clearInterval(progressUpdateId);
      setError(error instanceof Error ? error.message : 'Failed to process query');
      setLoading(false);
    }

    // Cleanup function
    return () => {
      if (pollInterval) clearInterval(pollInterval);
      if (timeoutId) clearTimeout(timeoutId);
      if (progressUpdateId) clearInterval(progressUpdateId);
    };
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