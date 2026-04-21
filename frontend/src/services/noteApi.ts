const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const noteApi = {
  fetchNotes: async (signal?: AbortSignal) => {
    const res = await fetch(`${API_BASE_URL}/api/notes/`, { signal });
    if (!res.ok) throw new Error('Failed to fetch notes');
    const data = await res.json();
    return data.notes || [];
  },

  fetchNoteContent: async (id: string, signal?: AbortSignal) => {
    const res = await fetch(`${API_BASE_URL}/api/notes/${id}`, { signal });
    if (!res.ok) throw new Error('Failed to fetch note content');
    const data = await res.json();
    return data.content || '';
  },

  // TODO: Add these endpoints for future scalability
  // createNote: async (data: { title: string; content: string }) => { ... }
  // updateNote: async (id: string, data: { title?: string; content?: string }) => { ... }
  // deleteNote: async (id: string) => { ... }
};
