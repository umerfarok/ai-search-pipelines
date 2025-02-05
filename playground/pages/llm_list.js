import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { MessageSquare, Bot, UserRound, Loader2 } from 'lucide-react';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

const ModelList = () => {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [chatLoading, setChatLoading] = useState(false);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/llm/models');
      const data = await response.json();
      setModels(data.data);
    } catch (error) {
      console.error('Error fetching models:', error);
    } finally {
      setLoading(false);
    }
  };

  const createSession = async (modelId) => {
    try {
      const response = await fetch('/api/llm/chat/session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: modelId,
          user_id: 'user123' // In practice, get from auth context
        }),
      });
      const data = await response.json();
      setSessionId(data.data.session_id);
      setMessages([]);
    } catch (error) {
      console.error('Error creating session:', error);
    }
  };

  const handleModelSelect = async (modelId) => {
    setSelectedModel(modelId);
    await createSession(modelId);
  };

  const sendMessage = async () => {
    if (!input.trim() || !sessionId) return;

    const userMessage = {
      content: input,
      role: 'user',
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setChatLoading(true);

    try {
      const response = await fetch('/api/llm/chat/message', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          message: userMessage.content
        }),
      });

      const data = await response.json();
      
      setMessages(prev => [...prev, {
        content: data.data.response,
        role: 'assistant',
        timestamp: new Date().toISOString()
      }]);
    } catch (error) {
      console.error('Error sending message:', error);
    } finally {
      setChatLoading(false);
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString();
  };

  const getModelStatus = (status) => {
    const statusColors = {
      completed: 'text-green-500',
      failed: 'text-red-500',
      processing: 'text-yellow-500',
      pending: 'text-blue-500'
    };
    return statusColors[status] || 'text-gray-500';
  };

  return (
    <div className="flex gap-4 h-[800px]">
      {/* Model List */}
      <Card className="w-1/3">
        <CardHeader>
          <CardTitle>Trained Models</CardTitle>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[700px] pr-4">
            {loading ? (
              <div className="flex justify-center">
                <Loader2 className="w-6 h-6 animate-spin" />
              </div>
            ) : (
              <div className="space-y-4">
                {models.map((model) => (
                  <Card 
                    key={model._id}
                    className={`cursor-pointer transition-colors hover:bg-gray-100 
                      ${selectedModel === model._id ? 'border-2 border-blue-500' : ''}`}
                    onClick={() => handleModelSelect(model._id)}
                  >
                    <CardContent className="p-4">
                      <div className="flex justify-between items-start">
                        <div>
                          <h3 className="font-medium">{model.base_model}</h3>
                          <p className="text-sm text-gray-500">
                            Version: {model.version}
                          </p>
                        </div>
                        <span className={`text-sm ${getModelStatus(model.status)}`}>
                          {model.status}
                        </span>
                      </div>
                      <div className="mt-2 text-sm text-gray-500">
                        Created: {formatDate(model.created_at)}
                      </div>
                      {model.training_stats && (
                        <div className="mt-2 text-sm">
                          <div>Progress: {model.training_stats.progress}%</div>
                          {model.training_stats.error && (
                            <div className="text-red-500">
                              Error: {model.training_stats.error}
                            </div>
                          )}
                        </div>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </ScrollArea>
        </CardContent>
      </Card>

      {/* Chat Interface */}
      <Card className="w-2/3">
        <CardHeader>
          <CardTitle>Chat</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col h-[700px]">
            <ScrollArea className="flex-grow mb-4 p-4 border rounded-lg">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex gap-2 mb-4 ${
                    message.role === 'user' ? 'justify-end' : 'justify-start'
                  }`}
                >
                  {message.role === 'assistant' && (
                    <Bot className="w-6 h-6 text-blue-500" />
                  )}
                  <div
                    className={`max-w-[80%] p-3 rounded-lg ${
                      message.role === 'user'
                        ? 'bg-blue-500 text-white'
                        : 'bg-gray-100'
                    }`}
                  >
                    {message.content}
                  </div>
                  {message.role === 'user' && (
                    <UserRound className="w-6 h-6 text-blue-500" />
                  )}
                </div>
              ))}
              {chatLoading && (
                <div className="flex justify-start gap-2">
                  <Bot className="w-6 h-6 text-blue-500" />
                  <div className="bg-gray-100 p-3 rounded-lg">
                    <Loader2 className="w-4 h-4 animate-spin" />
                  </div>
                </div>
              )}
            </ScrollArea>

            <div className="flex gap-2">
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={selectedModel ? "Type your message..." : "Select a model to start chatting"}
                disabled={!selectedModel || chatLoading}
                onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
              />
              <Button
                onClick={sendMessage}
                disabled={!selectedModel || !input.trim() || chatLoading}
              >
                {chatLoading ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <MessageSquare className="w-4 h-4" />
                )}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ModelList;