import { FileText, Plus, Search, Settings } from 'lucide-react';
import { Note } from '../types';

interface SidebarProps {
  notes: Note[];
  isLoadingNotes: boolean;
  activeTabId: string;
  handleOpenFile: (note: Note) => void;
  handleNewNote: () => void;
}

export default function Sidebar({ notes, isLoadingNotes, activeTabId, handleOpenFile, handleNewNote }: SidebarProps) {
  return (
    <div className="w-72 border-r border-gray-200 flex flex-col bg-gray-50/50 shrink-0">
      <div className="p-4 border-b border-gray-200 flex items-center justify-between shrink-0">
        <h1 className="font-bold text-lg text-gray-700 tracking-tight">Local KB</h1>
        <button 
          onClick={handleNewNote}
          className="p-1 hover:bg-gray-200 rounded text-gray-500 transition-colors"
        >
          <Plus size={18} />
        </button>
      </div>
      
      <div className="p-3 border-b border-gray-200">
        <div className="relative">
          <Search size={14} className="absolute left-2.5 top-2.5 text-gray-400" />
          <input 
            type="text" 
            placeholder="Search notes..." 
            className="w-full pl-8 pr-3 py-1.5 bg-white border border-gray-300 rounded-md text-sm focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
          />
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {isLoadingNotes ? (
          <div className="p-4 text-sm text-gray-500 text-center">Loading notes...</div>
        ) : notes.length === 0 ? (
          <div className="p-4 text-sm text-gray-500 text-center">No notes found.</div>
        ) : (
          notes.map((note) => (
            <div 
              key={note.id} 
              onClick={() => handleOpenFile(note)}
              className={`flex items-center gap-2 p-2 rounded-md cursor-pointer text-sm transition-colors
                ${activeTabId === note.id ? 'bg-blue-50 text-blue-700 font-medium' : 'hover:bg-gray-200 text-gray-700'}`}
            >
              <FileText size={16} className={activeTabId === note.id ? 'text-blue-500' : 'text-gray-400'} />
              <span className="truncate">{note.title}</span>
            </div>
          ))
        )}
      </div>

      <div className="p-3 border-t border-gray-200 flex items-center gap-2 text-sm text-gray-500 hover:bg-gray-200 hover:text-gray-800 cursor-pointer transition-colors">
        <Settings size={16} />
        <span>Settings</span>
      </div>
    </div>
  );
}
