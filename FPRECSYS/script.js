document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const loadingScreen = document.getElementById('loadingScreen');
    const mainContainer = document.getElementById('mainContainer');
    const chatInput = document.getElementById('chatInput');
    const sendChat = document.getElementById('sendChat');
    const chatMessages = document.getElementById('chatMessages');
    const recommendationsList = document.getElementById('recommendationsList');
    const playlistTracks = document.getElementById('playlistTracks');
    const playlistCount = document.getElementById('playlistCount');
    const searchInput = document.getElementById('searchInput');
    const searchBtn = document.getElementById('searchBtn');
    const searchResults = document.getElementById('searchResults');
    const clearPlaylist = document.getElementById('clearPlaylist');
    const exportCSV = document.getElementById('exportCSV');
    const moodButtons = document.querySelectorAll('.mood-btn');
    const statsTracks = document.getElementById('statsTracks');
    const statsPlaylist = document.getElementById('statsPlaylist');
    const statusIndicator = document.getElementById('statusIndicator');
    const statusText = document.getElementById('statusText');
    const toast = document.getElementById('toast');

    // State
    let playlist = [];
    let recommendations = [];

    // Initialize
    initializeApp();

    // Event Listeners
    sendChat.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

    searchBtn.addEventListener('click', performSearch);
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') performSearch();
    });

    clearPlaylist.addEventListener('click', clearPlaylistHandler);
    exportCSV.addEventListener('click', exportCSVHandler);

    moodButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const mood = btn.dataset.mood;
            sendMoodRequest(mood);
        });
    });

    // Functions
    async function initializeApp() {
        try {
            // Check if server is ready
            const response = await fetch('/api/stats');
            if (!response.ok) throw new Error('Server not ready');
            
            const stats = await response.json();
            
            // Initialize session
            const initResponse = await fetch('/api/initialize');
            const initData = await initResponse.json();
            
            // Update stats
            statsTracks.textContent = `${stats.total_tracks} tracks`;
            statsPlaylist.textContent = `${initData.initial_tracks.length} in playlist`;
            
            // Update playlist
            playlist = initData.initial_tracks.map(t => t.index);
            renderPlaylist(initData.initial_tracks);
            
            // Update status
            statusIndicator.classList.add('connected');
            statusText.textContent = 'Connected';
            
            // Show main interface
            setTimeout(() => {
                loadingScreen.style.display = 'none';
                mainContainer.style.display = 'block';
                showToast('System ready! Start chatting with the AI.');
            }, 1000);
            
        } catch (error) {
            console.error('Initialization failed:', error);
            statusText.textContent = 'Connection failed';
            showToast('Failed to connect to server. Please refresh.', 'error');
            
            // Fallback: Show interface anyway
            setTimeout(() => {
                loadingScreen.style.display = 'none';
                mainContainer.style.display = 'block';
            }, 2000);
        }
    }

    async function sendMessage() {
        const message = chatInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        addMessage(message, 'user');
        chatInput.value = '';
        
        // Show typing indicator
        const typingMsg = addMessage('Thinking...', 'bot');
        
        try {
            // Send to backend
            const response = await fetch('/api/recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mood: message, n: 10 })
            });
            
            const data = await response.json();
            
            // Remove typing indicator
            chatMessages.removeChild(typingMsg);
            
            // Add bot response
            addMessage(data.interpretation, 'bot');
            
            // Show recommendations
            recommendations = data.recommendations;
            renderRecommendations(recommendations);
            
        } catch (error) {
            console.error('Error:', error);
            chatMessages.removeChild(typingMsg);
            addMessage('Sorry, I encountered an error. Please try again.', 'bot');
        }
    }

    async function sendMoodRequest(mood) {
        // Add user message to chat
        addMessage(`Make it more ${mood}`, 'user');
        
        // Show typing indicator
        const typingMsg = addMessage('Thinking...', 'bot');
        
        try {
            const response = await fetch('/api/recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mood: mood, n: 10 })
            });
            
            const data = await response.json();
            
            // Remove typing indicator
            chatMessages.removeChild(typingMsg);
            
            // Add bot response
            addMessage(data.interpretation, 'bot');
            
            // Show recommendations
            recommendations = data.recommendations;
            renderRecommendations(recommendations);
            
        } catch (error) {
            console.error('Error:', error);
            chatMessages.removeChild(typingMsg);
            addMessage('Sorry, I encountered an error. Please try again.', 'bot');
        }
    }

    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        const prefix = sender === 'bot' ? '<strong>Playlist AI:</strong> ' : '<strong>You:</strong> ';
        contentDiv.innerHTML = prefix + text;
        
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        return messageDiv;
    }

    function renderRecommendations(tracks) {
        if (!tracks || tracks.length === 0) {
            recommendationsList.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-compass"></i>
                    <p>No recommendations found. Try a different mood!</p>
                </div>
            `;
            return;
        }
        
        recommendationsList.innerHTML = tracks.map((track, index) => `
            <div class="track-card" data-index="${track.index}">
                <div class="track-cover">
                    <i class="fas fa-music"></i>
                </div>
                <div class="track-info">
                    <div class="track-name">${track.name}</div>
                    <div class="track-artist">${track.artist}</div>
                    <div class="track-genre">${track.genre}</div>
                </div>
                <div class="track-actions">
                    <button class="btn-primary btn-small add-to-playlist" 
                            onclick="addTrackToPlaylist(${track.index})">
                        <i class="fas fa-plus"></i> Add
                    </button>
                </div>
            </div>
        `).join('');
    }

    async function addTrackToPlaylist(trackIndex) {
        try {
            const response = await fetch('/api/playlist', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ track_indices: [trackIndex] })
            });
            
            const data = await response.json();
            
            if (data.added_tracks && data.added_tracks.length > 0) {
                // Add to local playlist
                playlist.push(trackIndex);
                
                // Update playlist display
                const track = data.added_tracks[0];
                renderPlaylistItem(track, playlist.length);
                
                // Update count
                playlistCount.textContent = `${playlist.length} tracks`;
                statsPlaylist.textContent = `${playlist.length} in playlist`;
                
                // Show success
                showToast(`Added "${track.name}" to playlist`);
                
                // Remove from recommendations
                recommendations = recommendations.filter(t => t.index !== trackIndex);
                renderRecommendations(recommendations);
            }
            
        } catch (error) {
            console.error('Error adding track:', error);
            showToast('Failed to add track to playlist', 'error');
        }
    }

    function renderPlaylist(tracks) {
        playlistTracks.innerHTML = '';
        
        if (!tracks || tracks.length === 0) {
            playlistTracks.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-music"></i>
                    <p>Your playlist is empty. Add some tracks!</p>
                </div>
            `;
            return;
        }
        
        tracks.forEach((track, index) => {
            renderPlaylistItem(track, index + 1);
        });
        
        playlistCount.textContent = `${tracks.length} tracks`;
    }

    function renderPlaylistItem(track, number) {
        const trackDiv = document.createElement('div');
        trackDiv.className = 'playlist-track';
        trackDiv.dataset.index = track.index;
        
        trackDiv.innerHTML = `
            <div class="playlist-track-info">
                <div class="track-number">${number}</div>
                <div>
                    <div class="track-name">${track.name}</div>
                    <div class="track-artist">${track.artist}</div>
                </div>
            </div>
            <button class="btn-danger btn-small" onclick="removeFromPlaylist(${track.index})">
                <i class="fas fa-trash"></i>
            </button>
        `;
        
        playlistTracks.appendChild(trackDiv);
    }

    async function removeFromPlaylist(trackIndex) {
        try {
            const response = await fetch(`/api/playlist/${trackIndex}`, {
                method: 'DELETE'
            });
            
            const data = await response.json();
            
            if (data.removed_index) {
                // Remove from local playlist
                playlist = playlist.filter(idx => idx !== trackIndex);
                
                // Remove from DOM
                const trackElement = document.querySelector(`.playlist-track[data-index="${trackIndex}"]`);
                if (trackElement) {
                    trackElement.remove();
                }
                
                // Update count
                playlistCount.textContent = `${playlist.length} tracks`;
                statsPlaylist.textContent = `${playlist.length} in playlist`;
                
                // Update track numbers
                document.querySelectorAll('.playlist-track').forEach((el, index) => {
                    const numDiv = el.querySelector('.track-number');
                    if (numDiv) {
                        numDiv.textContent = index + 1;
                    }
                });
                
                showToast('Track removed from playlist');
            }
            
        } catch (error) {
            console.error('Error removing track:', error);
            showToast('Failed to remove track', 'error');
        }
    }

    async function performSearch() {
        const query = searchInput.value.trim();
        if (!query) return;
        
        try {
            const response = await fetch(`/api/search?q=${encodeURIComponent(query)}&limit=10`);
            const data = await response.json();
            
            if (data.results && data.results.length > 0) {
                searchResults.innerHTML = data.results.map(track => `
                    <div class="track-card" data-index="${track.index}">
                        <div class="track-cover">
                            <i class="fas fa-music"></i>
                        </div>
                        <div class="track-info">
                            <div class="track-name">${track.name}</div>
                            <div class="track-artist">${track.artist}</div>
                            <div class="track-genre">${track.genre}</div>
                        </div>
                        <div class="track-actions">
                            <button class="btn-primary btn-small" 
                                    onclick="addTrackToPlaylist(${track.index})">
                                <i class="fas fa-plus"></i> Add
                            </button>
                        </div>
                    </div>
                `).join('');
            } else {
                searchResults.innerHTML = `
                    <div class="empty-state">
                        <i class="fas fa-search"></i>
                        <p>No tracks found for "${query}"</p>
                    </div>
                `;
            }
            
        } catch (error) {
            console.error('Search error:', error);
            searchResults.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-exclamation-triangle"></i>
                    <p>Search failed. Please try again.</p>
                </div>
            `;
        }
    }

    async function clearPlaylistHandler() {
        if (!playlist.length) return;
        
        if (confirm('Are you sure you want to clear your playlist?')) {
            try {
                // Clear from server
                for (const trackIndex of playlist) {
                    await fetch(`/api/playlist/${trackIndex}`, { method: 'DELETE' });
                }
                
                // Clear locally
                playlist = [];
                playlistTracks.innerHTML = `
                    <div class="empty-state">
                        <i class="fas fa-music"></i>
                        <p>Your playlist is empty. Add some tracks!</p>
                    </div>
                `;
                
                playlistCount.textContent = '0 tracks';
                statsPlaylist.textContent = '0 in playlist';
                
                showToast('Playlist cleared');
                
            } catch (error) {
                console.error('Error clearing playlist:', error);
                showToast('Failed to clear playlist', 'error');
            }
        }
    }

    async function exportCSVHandler() {
        if (!playlist.length) {
            showToast('Playlist is empty', 'error');
            return;
        }
        
        try {
            const response = await fetch('/api/export/csv', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Create download link
                const link = document.createElement('a');
                link.href = data.download_url;
                link.download = data.filename;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                
                showToast(`Exported ${data.track_count} tracks to CSV`);
            } else {
                showToast(data.error || 'Export failed', 'error');
            }
            
        } catch (error) {
            console.error('Export error:', error);
            showToast('Export failed. Please try again.', 'error');
        }
    }

    function showToast(message, type = 'success') {
        toast.textContent = message;
        toast.className = 'toast';
        
        if (type === 'error') {
            toast.style.background = 'var(--danger)';
        } else if (type === 'warning') {
            toast.style.background = 'var(--warning)';
        } else {
            toast.style.background = 'var(--success)';
        }
        
        toast.classList.add('show');
        
        setTimeout(() => {
            toast.classList.remove('show');
        }, 3000);
    }

    // Make functions globally available for onclick handlers
    window.addTrackToPlaylist = addTrackToPlaylist;
    window.removeFromPlaylist = removeFromPlaylist;
});
