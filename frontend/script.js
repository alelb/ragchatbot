// API base URL - use relative path to work from any host
const API_URL = '/api';

// Global state
let currentSessionId = null;
let pendingQuery = null; // AbortController of the in-flight query, if any

// DOM elements
let chatMessages, chatInput, sendButton, newChatButton, totalCourses, courseTitles;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements after page loads
    chatMessages = document.getElementById('chatMessages');
    chatInput = document.getElementById('chatInput');
    sendButton = document.getElementById('sendButton');
    newChatButton = document.getElementById('newChatButton');
    totalCourses = document.getElementById('totalCourses');
    courseTitles = document.getElementById('courseTitles');
    
    setupEventListeners();
    createNewSession();
    loadCourseStats();
});

// Event Listeners
function setupEventListeners() {
    // Chat functionality
    sendButton.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

    // New chat
    newChatButton.addEventListener('click', startNewChat);
    
    // Suggested questions
    document.querySelectorAll('.suggested-item').forEach(button => {
        button.addEventListener('click', (e) => {
            const question = e.target.getAttribute('data-question');
            chatInput.value = question;
            sendMessage();
        });
    });
}


// Chat Functions
async function sendMessage() {
    const query = chatInput.value.trim();
    if (!query) return;

    // Disable input
    chatInput.value = '';
    chatInput.disabled = true;
    sendButton.disabled = true;

    // Add user message
    addMessage(query, 'user');

    // Add loading message - create a unique container for it
    const loadingMessage = createLoadingMessage();
    chatMessages.appendChild(loadingMessage);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    const controller = new AbortController();
    pendingQuery = controller;

    try {
        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                session_id: currentSessionId
            }),
            signal: controller.signal
        });

        if (!response.ok) throw new Error('Query failed');

        const data = await response.json();
        
        // The backend may hand out a new session (first query, or the old one was unknown)
        currentSessionId = data.session_id;

        // Replace loading message with response
        loadingMessage.remove();
        addMessage(data.answer, 'assistant', data.sources);

    } catch (error) {
        // Aborted by "New chat": the chat was already reset, nothing to show
        if (controller.signal.aborted) return;
        // Replace loading message with error
        loadingMessage.remove();
        addMessage(`Error: ${error.message}`, 'assistant');
    } finally {
        if (pendingQuery === controller) pendingQuery = null;
        chatInput.disabled = false;
        sendButton.disabled = false;
        chatInput.focus();
    }
}

function createLoadingMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.innerHTML = `
        <div class="message-content">
            <div class="loading">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    return messageDiv;
}

function addMessage(content, type, sources = null, isWelcome = false) {
    const messageId = Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}${isWelcome ? ' welcome-message' : ''}`;
    messageDiv.id = `message-${messageId}`;
    
    // Convert markdown to HTML for assistant messages
    const displayContent = type === 'assistant' ? marked.parse(content) : escapeHtml(content);
    
    let html = `<div class="message-content">${displayContent}</div>`;
    
    if (sources && sources.length > 0) {
        html += `
            <details class="sources-collapsible">
                <summary class="sources-header">Sources</summary>
                <div class="sources-content">${sources.map(renderSource).join('')}</div>
            </details>
        `;
    }
    
    messageDiv.innerHTML = html;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageId;
}

// Render a source as a chip linking to its lesson video (new tab), or a plain chip if it has no link
function renderSource(source) {
    // Source text is "<course title> - Lesson N"; show the lesson first, the course as secondary text
    const match = source.text.match(/^(.*) - (Lesson \d+)$/);
    const lesson = match ? match[2] : '';
    const course = match ? match[1] : source.text;
    const title = escapeHtml(source.text).replace(/"/g, '&quot;');
    const label = (lesson ? `<span class="source-lesson">${escapeHtml(lesson)}</span>` : '')
        + `<span class="source-course">${escapeHtml(course)}</span>`;

    if (!source.url || !/^https?:\/\//i.test(source.url)) {
        return `<span class="source-chip" title="${title}">${label}</span>`;
    }
    const href = escapeHtml(source.url).replace(/"/g, '&quot;');
    return `<a class="source-chip" href="${href}" target="_blank" rel="noopener noreferrer" title="${title}">`
        + `${label}<span class="source-icon" aria-hidden="true">↗</span></a>`;
}

// Helper function to escape HTML for user messages
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Removed removeMessage function - no longer needed since we handle loading differently

async function createNewSession() {
    currentSessionId = null;
    chatMessages.innerHTML = '';
    addMessage('Welcome to the Course Materials Assistant! I can help you with questions about courses, lessons and specific content. What would you like to know?', 'assistant', null, true);
}

// Discard the current conversation (UI and backend history) and start over without reloading
function startNewChat() {
    if (pendingQuery) pendingQuery.abort();

    const oldSessionId = currentSessionId;
    if (oldSessionId) {
        // Best-effort cleanup; the chat resets even if this fails
        fetch(`${API_URL}/session/${encodeURIComponent(oldSessionId)}`, { method: 'DELETE' })
            .catch(error => console.error('Error deleting session:', error));
    }

    createNewSession();
    chatInput.value = '';
    chatInput.disabled = false;
    sendButton.disabled = false;
    chatInput.focus();
}

// Load course statistics
async function loadCourseStats() {
    try {
        console.log('Loading course stats...');
        const response = await fetch(`${API_URL}/courses`);
        if (!response.ok) throw new Error('Failed to load course stats');
        
        const data = await response.json();
        console.log('Course data received:', data);
        
        // Update stats in UI
        if (totalCourses) {
            totalCourses.textContent = data.total_courses;
        }
        
        // Update course titles
        if (courseTitles) {
            if (data.course_titles && data.course_titles.length > 0) {
                courseTitles.innerHTML = data.course_titles
                    .map(title => `<div class="course-title-item">${title}</div>`)
                    .join('');
            } else {
                courseTitles.innerHTML = '<span class="no-courses">No courses available</span>';
            }
        }
        
    } catch (error) {
        console.error('Error loading course stats:', error);
        // Set default values on error
        if (totalCourses) {
            totalCourses.textContent = '0';
        }
        if (courseTitles) {
            courseTitles.innerHTML = '<span class="error">Failed to load courses</span>';
        }
    }
}