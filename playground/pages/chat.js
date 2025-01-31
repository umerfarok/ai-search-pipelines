import ChatBot from '../components/ChatBot';

export default function ChatPage() {
    return (
        <div className="min-h-screen bg-gray-100 p-8">
            <div className="max-w-4xl mx-auto">
                <h1 className="text-3xl font-bold mb-6">Product Search Assistant</h1>
                <p className="text-gray-600 mb-8">
                    Use our AI-powered chat assistant to find products and get detailed information.
                    Try asking questions like:
                </p>
                <div className="bg-white p-6 rounded-lg shadow-md">
                    <h2 className="text-xl font-semibold mb-4">Example Questions</h2>
                    <ul className="space-y-2 text-gray-700">
                        <li>"Show me products in the electronics category"</li>
                        <li>"What products do you have under $100?"</li>
                        <li>"Tell me about your best-selling items"</li>
                        <li>"Find products with high customer ratings"</li>
                    </ul>
                </div>
            </div>
            <ChatBot />
        </div>
    );
}
