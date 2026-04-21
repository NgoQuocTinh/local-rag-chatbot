import { MessageSquare, Send } from 'lucide-react';

export default function Chat() {
  return (
    <div className="w-80 lg:w-96 border-l border-gray-200 flex flex-col bg-gray-50/50 shrink-0">
      <div className="h-14 border-b border-gray-200 flex items-center px-4 gap-2 bg-white shrink-0">
        <MessageSquare size={18} className="text-blue-500" />
        <span className="font-semibold text-gray-700">AI Assistant</span>
      </div>
      
      {/* Chat History */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        <div className="bg-white p-3.5 rounded-lg border border-gray-200 shadow-sm text-sm text-gray-700 leading-relaxed">
          <p>Hi! I&apos;m your local AI. I have access to your knowledge base.</p>
          <p className="mt-2 text-gray-500 italic">How can I help you today?</p>
        </div>
        
        <div className="bg-blue-50 p-3.5 rounded-lg border border-blue-100 text-sm text-blue-900 leading-relaxed self-end w-fit max-w-[85%] ml-auto">
          Explain RAG to me in simple terms.
        </div>
      </div>

      {/* Chat Input Box */}
      <div className="p-4 bg-white border-t border-gray-200 shrink-0">
        <div className="relative">
          <textarea 
            rows={3}
            placeholder="Ask about your notes..." 
            className="w-full p-3 pr-10 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 resize-none"
          />
          <button className="absolute bottom-4 right-3 text-blue-500 hover:text-blue-700 bg-blue-50 p-1.5 rounded-md transition-colors">
            <Send size={16} />
          </button>
        </div>
      </div>
    </div>
  );
}
