import React, { useState, useCallback, useRef } from 'react';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Upload, MessageSquare, Loader2 } from 'lucide-react';

const LLMTrainingAndChat = () => {
  const [activeTab, setActiveTab] = useState('training');
  const [uploadedFile, setUploadedFile] = useState(null);
  const [modelConfig, setModelConfig] = useState({
    baseModel: 'mistralai/Mistral-7B-v0.1',
    batchSize: 4,
    epochs: 3,
    learningRate: 0.0002
  });
  const [messages, setMessages] = useState([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [isTraining, setIsTraining] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [sessionId, setSessionId] = useState('');
  const fileInputRef = useRef(null);

  const handleFileUpload = useCallback((event) => {
    const file = event.target.files[0];
    if (file) {
      setUploadedFile(file);
      setError('');
    }
  }, []);

  const handleTrainingSubmit = async () => {
    if (!uploadedFile) {
      setError('Please upload a document first');
      return;
    }

    setIsTraining(true);
    setError('');

    const formData = new FormData();
    formData.append('file', uploadedFile);
    formData.append('config', JSON.stringify({
      ...modelConfig,
      mode: 'create',
      training_config: {
        batch_size: parseInt(modelConfig.batchSize),
        epochs: parseInt(modelConfig.epochs),
        learning_rate: parseFloat(modelConfig.learningRate)
      },
      base_model: modelConfig.baseModel
    }));

    try {
      const response = await fetch('/api/config', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Training configuration failed');
      }

      const data = await response.json();
      setSessionId(data.data.config_id);
      setActiveTab('chat');
    } catch (err) {
      setError(err.message);
    } finally {
      setIsTraining(false);
    }
  };

  const handleSendMessage = async () => {
    if (!currentMessage.trim() || !sessionId) return;

    const newMessage = {
      content: currentMessage,
      role: 'user',
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, newMessage]);
    setCurrentMessage('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat/message', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          message: currentMessage
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to send message');
      }

      const data = await response.json();
      setMessages(prev => [...prev, {
        content: data.response,
        role: 'assistant',
        timestamp: new Date().toISOString()
      }]);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList>
            <TabsTrigger value="training">Training</TabsTrigger>
            <TabsTrigger value="chat">Chat</TabsTrigger>
          </TabsList>
        </Tabs>
      </CardHeader>

      <CardContent>
        {error && (
          <Alert variant="destructive" className="mb-4">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <TabsContent value="training" className="space-y-4">
          <div className="space-y-4">
            <div>
              <Input
                type="text"
                placeholder="Base Model"
                value={modelConfig.baseModel}
                onChange={(e) => setModelConfig(prev => ({ ...prev, baseModel: e.target.value }))}
                className="mb-2"
              />
            </div>

            <div className="grid grid-cols-3 gap-4">
              <Input
                type="number"
                placeholder="Batch Size"
                value={modelConfig.batchSize}
                onChange={(e) => setModelConfig(prev => ({ ...prev, batchSize: e.target.value }))}
              />
              <Input
                type="number"
                placeholder="Epochs"
                value={modelConfig.epochs}
                onChange={(e) => setModelConfig(prev => ({ ...prev, epochs: e.target.value }))}
              />
              <Input
                type="number"
                placeholder="Learning Rate"
                value={modelConfig.learningRate}
                onChange={(e) => setModelConfig(prev => ({ ...prev, learningRate: e.target.value }))}
                step="0.0001"
              />
            </div>

            <div className="flex items-center gap-4">
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileUpload}
                className="hidden"
                accept=".pdf,.doc,.docx,.txt"
              />
              <Button
                variant="outline"
                onClick={() => fileInputRef.current?.click()}
                className="w-full"
              >
                <Upload className="w-4 h-4 mr-2" />
                Upload Document
              </Button>
              {uploadedFile && <span className="text-sm">{uploadedFile.name}</span>}
            </div>

            <Button
              onClick={handleTrainingSubmit}
              disabled={isTraining || !uploadedFile}
              className="w-full"
            >
              {isTraining ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Training...
                </>
              ) : (
                'Start Training'
              )}
            </Button>
          </div>
        </TabsContent>

        <TabsContent value="chat" className="space-y-4">
          <div className="h-[400px] overflow-y-auto border rounded-lg p-4 space-y-4">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`flex ${
                  message.role === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                <div
                  className={`max-w-[70%] rounded-lg p-3 ${
                    message.role === 'user'
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-100'
                  }`}
                >
                  {message.content}
                </div>
              </div>
            ))}
          </div>

          <div className="flex gap-2">
            <Input
              value={currentMessage}
              onChange={(e) => setCurrentMessage(e.target.value)}
              placeholder="Type your message..."
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
            />
            <Button
              onClick={handleSendMessage}
              disabled={isLoading || !sessionId}
            >
              {isLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <MessageSquare className="w-4 h-4" />
              )}
            </Button>
          </div>
        </TabsContent>
      </CardContent>
    </Card>
  );
};

export default LLMTrainingAndChat;