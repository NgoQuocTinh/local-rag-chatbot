
'use client';
import { useState, useEffect, useRef } from 'react';
import Sidebar from '../components/Sidebar';
import Editor from '../components/Editor';
import Chat from '../components/Chat';
import { Note, Tab } from '../types';
import { noteApi } from '../services/noteApi';

export default function Home() {
  const [viewMode, setViewMode] = useState<'editor' | 'graph'>('editor');
  const [openTabs, setOpenTabs] = useState<Tab[]>([]);
  const [activeTabId, setActiveTabId] = useState<string>('');
  const [notes, setNotes] = useState<Note[]>([]);
  const [isLoadingNotes, setIsLoadingNotes] = useState(false);

  // UX Error state
  const [error, setError] = useState<string | null>(null);

  // Đổi tên state tabContents -> noteContentsById để tường minh hơn
  const [noteContentsById, setNoteContentsById] = useState<Record<string, string>>({});
  const [isLoadingContent, setIsLoadingContent] = useState(false);

  // Dùng để keep track file đã load (tránh reload nếu activeTabId đổi qua lại)
  const loadedNotesRef = useRef<Set<string>>(new Set());

  // Fetch file list
  useEffect(() => {
    const controller = new AbortController();

    const fetchAllNotes = async () => {
      setIsLoadingNotes(true);
      setError(null);
      try {
        const data = await noteApi.fetchNotes(controller.signal);
        setNotes(data || []);
      } catch (err: unknown) {
        if (err instanceof Error && err.name !== 'AbortError') {
          console.error("Error fetching notes:", err);
          setError("Failed to load notes. Please check connection.");
        }
      } finally {
        setIsLoadingNotes(false);
      }
    };

    fetchAllNotes();
    return () => controller.abort(); // Cleanup/Abort if tab switches very rapidly (though rare for notes list)
  }, []);

  // Lắng nghe sự thay đổi của activeTabId để lấy content từ Backend
  useEffect(() => {
    if (!activeTabId || activeTabId.startsWith('draft-') || loadedNotesRef.current.has(activeTabId)) {
      return; 
    }

    const controller = new AbortController();

    const fetchContent = async () => {
      setIsLoadingContent(true);
      setError(null);
      try {
        const content = await noteApi.fetchNoteContent(activeTabId, controller.signal);
        setNoteContentsById(prev => ({ ...prev, [activeTabId]: content }));
        loadedNotesRef.current.add(activeTabId);
      } catch (err: unknown) {
        if (err instanceof Error && err.name !== 'AbortError') {
          console.error("Error fetching note content:", err);
          setError(`Failed to load content for note.`);
        }
      } finally {
        setIsLoadingContent(false);
      }
    };

    fetchContent();

    return () => {
      controller.abort(); // Cancel pending request if activeTabId changes before this one finishes
    };
  }, [activeTabId]); // Dependency chỉ còn activeTabId

  // Hàm mở file sử dụng functional state update để tránh stale state
  const handleOpenFile = (note: Note) => {
    setOpenTabs(prevTabs => {
      if (!prevTabs.find(tab => tab.id === note.id)) {
        return [...prevTabs, { id: note.id, title: note.title }];
      }
      return prevTabs;
    });
    setActiveTabId(note.id);
  };

  // Hàm mở note nháp mới
  const handleNewNote = () => {
    const newId = `draft-${Date.now()}`;
    setOpenTabs(prevTabs => [...prevTabs, { id: newId, title: 'Untitled Note' }]);
    setNoteContentsById(prev => ({ ...prev, [newId]: "" }));
    loadedNotesRef.current.add(newId);
    setActiveTabId(newId);
  };

  // Hàm xử lý việc gõ text vào note
  const handleContentChange = (newContent: string) => {
    setNoteContentsById(prev => ({ ...prev, [activeTabId]: newContent }));
  };

  // Hàm đóng tab
  const handleCloseTab = (e: React.MouseEvent, idToClose: string) => {
    e.stopPropagation();
    setOpenTabs(prevTabs => {
      const newTabs = prevTabs.filter(tab => tab.id !== idToClose);
      if (activeTabId === idToClose) {
        setActiveTabId(newTabs.length > 0 ? newTabs[newTabs.length - 1].id : '');
      }
      return newTabs;
    });
  };

  return (
    <div className="relative h-screen w-full flex bg-white text-gray-800 font-sans overflow-hidden">
      {error && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-50 bg-red-50 text-red-600 px-4 py-2 rounded-md shadow-sm border border-red-200 flex items-center gap-3">
          <span className="text-sm font-medium">{error}</span>
          <button onClick={() => setError(null)} className="text-red-400 hover:text-red-700">&times;</button>
        </div>
      )}

      <Sidebar 
        notes={notes}
        isLoadingNotes={isLoadingNotes}
        activeTabId={activeTabId}
        handleOpenFile={handleOpenFile}
        handleNewNote={handleNewNote}
      />
      
      <Editor 
        openTabs={openTabs}
        activeTabId={activeTabId}
        setActiveTabId={setActiveTabId}
        handleCloseTab={handleCloseTab}
        handleNewNote={handleNewNote}
        viewMode={viewMode}
        setViewMode={setViewMode}
        isLoadingContent={isLoadingContent}
        tabContents={noteContentsById}  // Prop component vẫn giữ nguyên
        handleContentChange={handleContentChange}
      />

      <Chat />
    </div>
  );
}

