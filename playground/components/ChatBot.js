import React, { useState, useRef, useEffect } from 'react';
import { MessageSquare, Send, Loader2, X, Maximize2, Minimize2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

// Add these new constants at the top
const BASIC_RESPONSES = {
    greeting: [
        "Hello! How can I help you find products today?",
        "Hi there! Looking for something specific?",
        "Welcome! What kind of products are you interested in?"
    ],
    farewell: [
        "Thank you for chatting! Let me know if you need anything else.",
        "Have a great day! Feel free to come back if you need more help.",
        "Goodbye! Don't hesitate to ask if you have more questions."
    ],
    unknown: [
        "I'm not sure about that. Would you like to search our products?",
        "I might need more details to help you better. Could you be more specific?",
        "I'm still learning, but I can help you search for products if you'd like."
    ]
};

const QUICK_RESPONSES = [
    "Show me the latest products",
    "What's on sale?",
    "Find popular items",
    "Browse by category"
];

const BasicPatterns = [
    {
        pattern: /(hi|hello|hey|greetings)/i,
        type: 'greeting'
    },
    {
        pattern: /(bye|goodbye|thank you|thanks|cya)/i,
        type: 'farewell'
    },
    {
        pattern: /(help|assist|support)/i,
        response: "I can help you find products, check prices, and provide product information. What would you like to know?"
    }
];

const ChatMessage = ({ message, isBot }) => (
    <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className={`flex ${isBot ? 'justify-start' : 'justify-end'} mb-4`}
    >
        <div
            className={`max-w-[80%] rounded-lg p-3 ${
                isBot
                    ? 'bg-gray-100 text-gray-800'
                    : 'bg-blue-500 text-white'
            }`}
        >
            {message.type === 'markdown' ? (
                <ReactMarkdown 
                    className="prose prose-sm max-w-none"
                    components={{
                        a: ({ node, ...props }) => (
                            <a {...props} className="text-blue-300 hover:text-blue-200" />
                        ),
                        code: ({ node, ...props }) => (
                            <code {...props} className="bg-gray-700 rounded px-1" />
                        ),
                    }}
                >
                    {message.content}
                </ReactMarkdown>
            ) : (
                <p className="whitespace-pre-wrap">{message.content}</p>
            )}
        </div>
    </motion.div>
);

const TypingIndicator = () => (
    <div className="flex items-center space-x-2 text-gray-400">
        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
    </div>
);

const ChatBot = ({ configId = 'latest' }) => {
    const [isOpen, setIsOpen] = useState(false);
    const [isMinimized, setIsMinimized] = useState(false);
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [isTyping, setIsTyping] = useState(false);
    const [quickResponses, setQuickResponses] = useState(QUICK_RESPONSES);
    const [showQuickResponses, setShowQuickResponses] = useState(true);
    const messagesEndRef = useRef(null);
    const inputRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleOpen = () => {
        setIsOpen(true);
        setIsMinimized(false);
        // Add initial greeting
        if (messages.length === 0) {
            setMessages([
                {
                    content: "Hello! I'm your AI shopping assistant. How can I help you find products today?",
                    type: 'text',
                    isBot: true
                }
            ]);
        }
    };

    const handleClose = () => {
        setIsOpen(false);
        setIsMinimized(false);
    };

    const handleToggleMinimize = () => {
        setIsMinimized(!isMinimized);
    };

    const getRandomResponse = (type) => {
        const responses = BASIC_RESPONSES[type];
        return responses[Math.floor(Math.random() * responses.length)];
    };

    const handleBasicChat = (input) => {
        for (const pattern of BasicPatterns) {
            if (pattern.pattern.test(input)) {
                if (pattern.type) {
                    return getRandomResponse(pattern.type);
                }
                return pattern.response;
            }
        }
        return null;
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!input.trim()) return;

        // Add user message
        const userMessage = { content: input.trim(), type: 'text', isBot: false };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsTyping(true);

        // Check for basic chat patterns first
        const basicResponse = handleBasicChat(input.trim());
        if (basicResponse) {
            setTimeout(() => {
                setMessages(prev => [...prev, {
                    content: basicResponse,
                    type: 'text',
                    isBot: true
                }]);
                setIsTyping(false);
            }, 500);
            return;
        }

        // Proceed with API call for product search
        try {
            const response = await fetch(`${API_BASE_URL}/search`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: input.trim(),
                    config_id: configId,
                    max_items: 5
                })
            });

            const data = await response.json();
            if (response.ok) {
                // Format bot response with markdown
                let botResponse = "Here's what I found:\n\n";
                
                if (data.natural_response) {
                    botResponse = data.natural_response + "\n\n";
                }

                if (data.results && data.results.length > 0) {
                    data.results.forEach((result, index) => {
                        botResponse += `**${result.name}**\n`;
                        botResponse += `- ${result.description}\n`;
                        botResponse += `- Category: ${result.category}\n`;
                        if (result.metadata) {
                            Object.entries(result.metadata).forEach(([key, value]) => {
                                botResponse += `- ${key}: ${value}\n`;
                            });
                        }
                        botResponse += `- Match Score: ${(result.score * 100).toFixed(1)}%\n\n`;
                    });
                } else {
                    botResponse += "I couldn't find any matching products. Try rephrasing your search.";
                }

                setMessages(prev => [...prev, { 
                    content: botResponse, 
                    type: 'markdown',
                    isBot: true 
                }]);
            } else {
                throw new Error(data.error || 'Search failed');
            }
        } catch (error) {
            setMessages(prev => [...prev, { 
                content: getRandomResponse('unknown'),
                type: 'text',
                isBot: true 
            }]);
        } finally {
            setIsTyping(false);
        }
    };

    const handleQuickResponse = (response) => {
        setInput(response);
        handleSubmit({ preventDefault: () => {} });
        setShowQuickResponses(false);
    };

    if (!isOpen) {
        return (
            <button
                onClick={handleOpen}
                className="fixed bottom-4 right-4 bg-blue-500 text-white p-4 rounded-full shadow-lg hover:bg-blue-600 transition-colors"
                aria-label="Open chat"
            >
                <MessageSquare className="h-6 w-6" />
            </button>
        );
    }

    return (
        <div
            className={`fixed bottom-4 right-4 w-96 bg-white rounded-lg shadow-xl transition-all duration-300 ${
                isMinimized ? 'h-14' : 'h-[600px]'
            }`}
        >
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b bg-blue-500 text-white rounded-t-lg">
                <h3 className="font-semibold">Shopping Assistant</h3>
                <div className="flex items-center space-x-2">
                    <button
                        onClick={handleToggleMinimize}
                        className="p-1 hover:bg-blue-600 rounded"
                        aria-label={isMinimized ? "Maximize chat" : "Minimize chat"}
                    >
                        {isMinimized ? (
                            <Maximize2 className="h-4 w-4" />
                        ) : (
                            <Minimize2 className="h-4 w-4" />
                        )}
                    </button>
                    <button
                        onClick={handleClose}
                        className="p-1 hover:bg-blue-600 rounded"
                        aria-label="Close chat"
                    >
                        <X className="h-4 w-4" />
                    </button>
                </div>
            </div>

            {/* Chat Messages */}
            <AnimatePresence>
                {!isMinimized && (
                    <>
                        <div className="flex-1 p-4 space-y-4 overflow-y-auto h-[calc(600px-8rem)]">
                            {messages.map((message, index) => (
                                <ChatMessage key={index} message={message} isBot={message.isBot} />
                            ))}
                            {isTyping && (
                                <div className="flex justify-start mb-4">
                                    <div className="bg-gray-100 rounded-lg p-3">
                                        <TypingIndicator />
                                    </div>
                                </div>
                            )}
                            {showQuickResponses && messages.length <= 1 && (
                                <div className="grid grid-cols-2 gap-2 mt-4">
                                    {quickResponses.map((response, index) => (
                                        <button
                                            key={index}
                                            onClick={() => handleQuickResponse(response)}
                                            className="p-2 text-sm text-left text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors"
                                        >
                                            {response}
                                        </button>
                                    ))}
                                </div>
                            )}
                            <div ref={messagesEndRef} />
                        </div>

                        {/* Input Form */}
                        <form
                            onSubmit={handleSubmit}
                            className="border-t p-4 bg-gray-50 rounded-b-lg"
                        >
                            <div className="flex space-x-2">
                                <input
                                    ref={inputRef}
                                    type="text"
                                    value={input}
                                    onChange={(e) => setInput(e.target.value)}
                                    placeholder="Ask about products..."
                                    className="flex-1 p-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                                    disabled={isTyping}
                                />
                                <button
                                    type="submit"
                                    disabled={isTyping || !input.trim()}
                                    className="bg-blue-500 text-white p-2 rounded-lg hover:bg-blue-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    {isTyping ? (
                                        <Loader2 className="h-5 w-5 animate-spin" />
                                    ) : (
                                        <Send className="h-5 w-5" />
                                    )}
                                </button>
                            </div>
                        </form>
                    </>
                )}
            </AnimatePresence>
        </div>
    );
};

export default ChatBot;
