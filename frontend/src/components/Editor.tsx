import { FileText, Network, X } from 'lucide-react';
import { Tab } from '../types';
import GraphView from './GraphView';

interface EditorProps {
  openTabs: Tab[];
  activeTabId: string;
  setActiveTabId: (id: string) => void;
  handleCloseTab: (e: React.MouseEvent, id: string) => void;
  handleNewNote: () => void;
  viewMode: 'editor' | 'graph';
  setViewMode: (mode: 'editor' | 'graph') => void;
  isLoadingContent: boolean;
  tabContents: Record<string, string>;
  handleContentChange: (content: string) => void;
  handleSaveNote: () => void;
  handleDeleteNote: () => void;
  handleTitleChange: (id: string, newTitle: string) => void;
  handleOpenFileFromGraph: (id: string) => void;
  isSaving: boolean;
  isSyncing: boolean;
}

export default function Editor(props: EditorProps) {
  const {
    openTabs,
    activeTabId,
    setActiveTabId,
    handleCloseTab,
    handleNewNote,
    viewMode,
    setViewMode,
    isLoadingContent,
    tabContents,
    handleContentChange,
    handleSaveNote,
    handleDeleteNote,
    handleTitleChange,
    handleOpenFileFromGraph,
    isSaving,
    isSyncing
  } = props;

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // ... (previous handleKeyDown code)

    const target = e.target as HTMLTextAreaElement;
    const value = target.value;
    const cursorPosition = target.selectionStart;

    // Get the content of the current line (from the previous \n to the cursor)
    const lastNewline = value.lastIndexOf('\n', cursorPosition - 1);
    const lineStart = lastNewline === -1 ? 0 : lastNewline + 1;
    const currentLine = value.substring(lineStart, cursorPosition);

    if (e.key === ' ') {
      // Auto-format bullet list: '-' or '*' + Space -> '• '
      if (currentLine === '-' || currentLine === '*') {
        e.preventDefault();
        const newValue = value.substring(0, lineStart) + '• ' + value.substring(cursorPosition);
        handleContentChange(newValue);
        
        // Restore cursor position after render
        setTimeout(() => {
          target.selectionStart = target.selectionEnd = lineStart + 2;
        }, 0);
      }
    } else if (e.key === 'Enter') {
      const numberMatch = currentLine.match(/^(\s*)(\d+)\.\s/);
      
      // Handle Bullet Lists
      if (currentLine.startsWith('• ')) {
        // Exit list if the line only contains the bullet
        if (currentLine.trim() === '•') {
          e.preventDefault();
          const newValue = value.substring(0, lineStart) + value.substring(cursorPosition);
          handleContentChange(newValue);
          setTimeout(() => {
            target.selectionStart = target.selectionEnd = lineStart;
          }, 0);
        } else {
          // Auto-continue bullet list
          e.preventDefault();
          const newValue = value.substring(0, cursorPosition) + '\n• ' + value.substring(cursorPosition);
          handleContentChange(newValue);
          setTimeout(() => {
            target.selectionStart = target.selectionEnd = cursorPosition + 3;
          }, 0);
        }
      } 
      // Handle Numbered Lists
      else if (numberMatch) {
        const indent = numberMatch[1];
        const num = parseInt(numberMatch[2], 10);
        // Exit list if the line only contains the number
        if (currentLine.trim() === `${num}.`) {
          e.preventDefault();
          const newValue = value.substring(0, lineStart) + value.substring(cursorPosition);
          handleContentChange(newValue);
          setTimeout(() => {
            target.selectionStart = target.selectionEnd = lineStart;
          }, 0);
        } else {
          // Auto-continue numbered list with incremented number
          e.preventDefault();
          const prefix = `\n${indent}${num + 1}. `;
          const newValue = value.substring(0, cursorPosition) + prefix + value.substring(cursorPosition);
          handleContentChange(newValue);
          setTimeout(() => {
            target.selectionStart = target.selectionEnd = cursorPosition + prefix.length;
          }, 0);
        }
      }
      // Handle Blockquotes
      else if (currentLine.startsWith('> ')) {
        // Exit blockquote if empty
        if (currentLine.trim() === '>') {
          e.preventDefault();
          const newValue = value.substring(0, lineStart) + value.substring(cursorPosition);
          handleContentChange(newValue);
          setTimeout(() => {
            target.selectionStart = target.selectionEnd = lineStart;
          }, 0);
        } else {
          // Auto-continue blockquote
          e.preventDefault();
          const newValue = value.substring(0, cursorPosition) + '\n> ' + value.substring(cursorPosition);
          handleContentChange(newValue);
          setTimeout(() => {
            target.selectionStart = target.selectionEnd = cursorPosition + 3;
          }, 0);
        }
      }
    } else if (e.key === 'Tab') {
      // Insert two spaces instead of switching focus
      e.preventDefault();
      const newValue = value.substring(0, cursorPosition) + '  ' + value.substring(cursorPosition);
      handleContentChange(newValue);
      setTimeout(() => {
        target.selectionStart = target.selectionEnd = cursorPosition + 2;
      }, 0);
    }
  };

  return (
    <div className="flex-1 flex flex-col min-w-0 bg-gray-50/30">
      {/* Top Header of Main Area (Tabs Bar) */}
      <div className="h-10 border-b border-gray-200 flex bg-gray-100/50 shrink-0 overflow-x-auto overflow-y-hidden">
        {openTabs.map(tab => (
          <div 
            key={tab.id}
            onClick={() => setActiveTabId(tab.id)}
            className={`group flex items-center gap-1.5 px-3 min-w-32 max-w-xs border-r border-gray-200 cursor-pointer transition-colors
              ${activeTabId === tab.id ? 'bg-white border-t-2 border-t-blue-500 text-gray-800' : 'bg-transparent border-t-2 border-t-transparent text-gray-500 hover:bg-gray-100'}
            `}
          >
              <FileText size={14} className={activeTabId === tab.id ? 'text-blue-500' : 'text-gray-400'} />
              <span className="text-sm truncate select-none flex-1 font-medium">{tab.title}</span>
              <button 
                onClick={(e) => handleCloseTab(e, tab.id)}
                className={`p-0.5 rounded hover:bg-gray-200 ${activeTabId === tab.id ? 'text-gray-400' : 'text-transparent group-hover:text-gray-400'} transition-all`}
              >
                <X size={14} />
              </button>
          </div>
        ))}
      </div>

      {/* Action Toolbar */}
      <div className="h-12 border-b border-gray-200 flex items-center px-4 justify-between bg-white shrink-0 shadow-sm z-10">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-gray-700">
            {openTabs.find(t => t.id === activeTabId)?.title || "No file selected"}
          </span>
        </div>
        
        {/* Tabs: Editor vs Graph */}
        <div className="flex bg-gray-100 p-1 rounded-lg">
          <button 
            onClick={() => setViewMode('editor')}
            className={`px-3 py-1 text-sm rounded-md transition-colors ${viewMode === 'editor' ? 'bg-white shadow-sm text-blue-600 font-medium' : 'text-gray-500 hover:text-gray-700'}`}
          >
            Editor
          </button>
          <button 
            onClick={() => setViewMode('graph')}
            className={`px-3 py-1 flex items-center gap-1.5 text-sm rounded-md transition-colors ${viewMode === 'graph' ? 'bg-white shadow-sm text-blue-600 font-medium' : 'text-gray-500 hover:text-gray-700'}`}
          >
            <Network size={14} /> Graph
          </button>
        </div>
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-auto bg-white">
        {viewMode === 'graph' ? (
          <div className="w-full h-full flex flex-col relative bg-gray-50 border-t border-gray-100 overflow-hidden">
             <div className="absolute top-2 left-4 z-10 text-sm text-gray-500 font-medium">
               Knowledge Graph
             </div>
             <GraphView onNodeClick={handleOpenFileFromGraph} />
          </div>
        ) : openTabs.length === 0 ? (
          <div className="w-full h-full flex flex-col items-center justify-center text-gray-400 gap-4 bg-gray-50">
            <FileText size={48} className="text-gray-300" />
            <p>No file is open.</p>
            <button onClick={handleNewNote} className="mt-2 px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition">Create New Note</button>
          </div>
        ) : (
          <div className="h-full flex flex-col mx-auto w-full max-w-4xl p-6 lg:p-10">
            <div className="flex items-center justify-between mb-6">
              <input
                className="flex-1 text-[18px] font-bold text-gray-900 tracking-tight px-1 outline-none focus:ring-2 focus:ring-blue-100 rounded-md transition-all bg-transparent disabled:opacity-75"
                value={openTabs.find(t => t.id === activeTabId)?.title || ''}
                onChange={(e) => handleTitleChange(activeTabId, e.target.value)}
                placeholder="Note Title"
              />
              <div className="flex gap-2 shrink-0">
                <button 
                  onClick={handleDeleteNote}
                  className="px-3 py-1.5 text-sm text-red-600 bg-red-50 hover:bg-red-100 rounded-md transition"
                >
                  Delete
                </button>
                <button 
                  onClick={handleSaveNote}
                  disabled={isSaving || isSyncing}
                  className="px-4 py-1.5 text-sm text-white bg-blue-500 hover:bg-blue-600 disabled:bg-blue-300 disabled:cursor-not-allowed rounded-md transition flex items-center gap-1.5"
                >
                  {isSaving ? 'Saving...' : isSyncing ? 'Syncing...' : 'Save & Sync'}
                </button>
              </div>
            </div>
            {isLoadingContent ? (
              <div className="text-gray-400 flex items-center justify-center flex-1">Loading content...</div>
            ) : (
              <textarea
                className="flex-1 w-full h-full resize-none p-1 outline-none text-gray-700 leading-relaxed text-[14px] bg-transparent border-none"
                value={tabContents[activeTabId] ?? ''}
                onChange={(e) => handleContentChange(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Start typing your note here..."
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}
