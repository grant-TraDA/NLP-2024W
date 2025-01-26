let lastUserInput = "";
let lastLLMResponse = "";
let currentConversationId = null;
let lastMessageId = null;
let ttsEnabled = localStorage.getItem('ttsEnabled') === 'true' || false;

// Initialize conversation ID when the page loads
document.addEventListener('DOMContentLoaded', function() {
    // Generate a new conversation ID if one doesn't exist
    if (!currentConversationId) {
        currentConversationId = 'conv_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        console.log('Generated new conversation ID:', currentConversationId);
    }

    // Initialize TTS toggle button state
    const ttsButton = document.getElementById('tts-toggle');
    if (ttsButton) {
        ttsButton.innerHTML = ttsEnabled ? '🔊 TTS On' : '🔈 TTS Off';
        ttsButton.classList.toggle('active', ttsEnabled);
    }
});

async function generateResponse() {
    const userInput = document.getElementById("user-input").value;
    if (!userInput) return alert("Please enter a message.");

    // Hide any previous feedback sections when new message is sent
    hidePreviousFeedback();

    // Ensure we have a conversation ID
    if (!currentConversationId) {
        currentConversationId = 'conv_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    lastUserInput = userInput;
    lastMessageId = 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    
    // Update UI immediately to show user message
    updateChat('user', userInput);
    
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                message: userInput,
                conversationId: currentConversationId,
                messageId: lastMessageId
            })
        });
        
        const data = await response.json();
        console.log('Received data from server:', data);

        if (data.response) {
            lastLLMResponse = data.response;
            // Use server-provided IDs if available, otherwise keep the generated ones
            lastMessageId = data.messageId || lastMessageId;
            currentConversationId = data.conversationId || currentConversationId;
            
            console.log('Using IDs:', {
                messageId: lastMessageId,
                conversationId: currentConversationId
            });
            
            updateChat('assistant', data.response);
        } else {
            // For errors, we still want to track the conversation
            updateChat('system', "Error: " + (data.error || "Unable to generate response"));
        }
        
        document.getElementById("user-input").value = "";
        scrollToBottom();
        
    } catch (error) {
        updateChat('system', "Error: Unable to generate response");
        console.error(error);
    }
}

function updateChat(role, content) {
    const chatDiv = document.getElementById("chat");
    const messageContainer = document.createElement("div");
    messageContainer.className = `message-container ${role}-container`;
    
    // Create message div
    const newMessage = document.createElement("div");
    newMessage.className = `chat-message ${role}`;
    
    // Add speech button for assistant messages
    if (role === 'assistant') {
        const messageContent = `
            <div class="message-header" style="margin-bottom: 8px;">
                <strong>${role.charAt(0).toUpperCase() + role.slice(1)}</strong>
                <button class="speak-button" 
                        onclick="handleSpeak(this, this.parentElement.nextElementSibling.textContent)" 
                        title="Listen"
                        style="display: ${ttsEnabled ? 'inline-flex' : 'none'}; margin-left: 8px;">🔊</button>
            </div>
            <div class="message-text">${content}</div>
        `;
        newMessage.innerHTML = messageContent;

        // Auto-play if TTS is enabled
        if (ttsEnabled) {
            requestAnimationFrame(() => {
                const textContent = newMessage.querySelector('.message-text').textContent;
                handleSpeak(null, textContent);
            });
        }
    } else {
        newMessage.innerHTML = `
            <div class="message-header" style="margin-bottom: 8px;">
                <strong>${role.charAt(0).toUpperCase() + role.slice(1)}</strong>
            </div>
            <div class="message-text">${content}</div>
        `;
    }
    
    messageContainer.appendChild(newMessage);
    
    // Add feedback section for both assistant responses and system errors
    if (role === 'assistant' || role === 'system') {
        const feedbackSection = createFeedbackSection();
        feedbackSection.dataset.messageId = lastMessageId;
        feedbackSection.dataset.conversationId = currentConversationId;
        messageContainer.appendChild(feedbackSection);
    }
    
    chatDiv.appendChild(messageContainer);
    scrollToBottom();
}

function createFeedbackSection() {
    const feedbackSection = document.createElement("div");
    feedbackSection.className = "feedback-section";
    
    const content = `
        <h4>Was this response helpful?</h4>
        <div class="feedback-buttons">
            <button onclick="handleFeedback(this, 'thumbs_up')" class="feedback-btn thumbs-up">👍 Yes</button>
            <button onclick="handleFeedback(this, 'thumbs_down')" class="feedback-btn thumbs-down">👎 No</button>
        </div>
        <div class="feedback-form" style="display: none;">
            <textarea placeholder="Please describe what would have been a better response... (optional)"></textarea>
            <button class="submit-feedback-btn">Submit Feedback</button>
        </div>
        <div class="feedback-success" style="display: none;">
            Thank you for your feedback!
        </div>
    `;
    
    feedbackSection.innerHTML = content;
    feedbackSection.style.display = 'block';

    // Add Enter key handler for feedback submission
    const textarea = feedbackSection.querySelector('textarea');
    textarea.addEventListener('keydown', function(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            const submitBtn = feedbackSection.querySelector('.submit-feedback-btn');
            submitBtn.click();
        }
    });
    
    return feedbackSection;
}

function handleFeedback(button, type) {
    const feedbackSection = button.closest('.feedback-section');
    const buttons = feedbackSection.querySelectorAll('.feedback-btn');
    const feedbackForm = feedbackSection.querySelector('.feedback-form');
    
    // Check if feedback was already given
    if (buttons[0].disabled) {
        return;
    }
    
    // Remove selected class and enable all buttons first
    buttons.forEach(btn => {
        btn.classList.remove('selected');
        btn.disabled = false;
    });
    
    // Add selected class to clicked button
    button.classList.add('selected');
    button.disabled = true;
    
    if (type === 'thumbs_down') {
        feedbackForm.style.display = 'block';
        
        // Add submit event listener
        const submitBtn = feedbackForm.querySelector('.submit-feedback-btn');
        submitBtn.onclick = () => submitDetailedFeedback(feedbackSection, type);
    } else {
        // Hide feedback form if it was previously shown
        if (feedbackForm) {
            feedbackForm.style.display = 'none';
        }
        submitDetailedFeedback(feedbackSection, type);
    }
}

async function submitDetailedFeedback(feedbackSection, type) {
    const feedbackForm = feedbackSection.querySelector('.feedback-form');
    const comment = feedbackForm ? feedbackForm.querySelector('textarea').value : '';
    
    // Get IDs from the feedback section's data attributes
    const messageId = feedbackSection.dataset.messageId;
    const conversationId = feedbackSection.dataset.conversationId;
    
    console.log('Submitting feedback with IDs:', {
        messageId: messageId,
        conversationId: conversationId,
        type: type
    });

    // Validate required data
    if (!messageId || !conversationId) {
        console.error('Missing required IDs:', { messageId, conversationId });
        alert('Missing required feedback data. Please try again.');
        return;
    }

    try {
        const response = await fetch('/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({
                conversationId: conversationId,
                messageId: messageId,
                type: type,
                comment: comment
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            if (feedbackForm) {
                feedbackForm.style.display = 'none';
            }
            
            const successDiv = feedbackSection.querySelector('.feedback-success');
            successDiv.style.display = 'block';
            
            setTimeout(() => {
                feedbackSection.style.display = 'none';
            }, 1500);
        } else {
            throw new Error(data.error || 'Failed to submit feedback');
        }
    } catch (error) {
        console.error('Error submitting feedback:', error);
        alert('Failed to submit feedback: ' + error.message);
        
        const buttons = feedbackSection.querySelectorAll('.feedback-btn');
        buttons.forEach(btn => {
            btn.disabled = false;
            btn.style.opacity = '1';
        });
    }
}

function scrollToBottom() {
    const chatContainer = document.getElementById("chat-container");
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function showFeedbackBox() {
    document.getElementById("feedback-box").style.display = "block";
}

async function submitFeedback(feedbackType) {
    const userFeedback = document.getElementById("user-feedback").value || "N/A";

    const response = await fetch('/submit_feedback', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            user_input: lastUserInput,
            llm_response: lastLLMResponse,
            feedback_type: feedbackType,
            user_feedback: feedbackType === 'thumbs_down' ? userFeedback : "N/A"
        })
    });

    const data = await response.json();
    if (data.status) {
        alert(data.status);
        document.getElementById("feedback-container").style.display = "none";
        document.getElementById("feedback-box").style.display = "none";
        document.getElementById("user-feedback").value = "";
    }
}

// Add event listener for Enter key
document.getElementById("user-input").addEventListener("keypress", function(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        generateResponse();
    }
}); 

// Add CSS styles programmatically for the disabled state
const style = document.createElement('style');
style.textContent = `
    .feedback-btn {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        min-width: 80px;  /* Ensure enough space for text */
    }
`;
document.head.appendChild(style);

// Add new function to handle conversation reset
function resetConversation() {
    // Clear chat container
    const chatDiv = document.getElementById("chat");
    chatDiv.innerHTML = '';
    
    // Generate new conversation ID
    currentConversationId = 'conv_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    lastMessageId = null;
    lastUserInput = "";
    lastLLMResponse = "";
    
    console.log('Started new conversation:', currentConversationId);
}

// Add theme switching functionality
function toggleTheme() {
    const body = document.body;
    const currentTheme = body.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    body.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    
    // Update button icon/text
    const themeButton = document.getElementById('theme-toggle');
    themeButton.innerHTML = newTheme === 'dark' ? '☀️' : '🌙';
}

// Initialize theme on page load
document.addEventListener('DOMContentLoaded', function() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.body.setAttribute('data-theme', savedTheme);
    
    // Initialize theme button
    const themeButton = document.getElementById('theme-toggle');
    if (themeButton) {
        themeButton.innerHTML = savedTheme === 'dark' ? '☀️' : '🌙';
    }
});

// Add this function to hide previous feedback sections
function hidePreviousFeedback() {
    const feedbackSections = document.querySelectorAll('.feedback-section');
    feedbackSections.forEach(section => {
        section.style.display = 'none';
    });
}

// Update the keyboard event listener to also handle new messages
document.addEventListener('keydown', function(event) {
    if (event.key === 'Enter' && !event.shiftKey && 
        event.target.id === 'user-input') {
        event.preventDefault();
        hidePreviousFeedback();
        generateResponse();
    }
});

// Add function to toggle TTS
function toggleTTS() {
    ttsEnabled = !ttsEnabled;
    localStorage.setItem('ttsEnabled', ttsEnabled);
    
    // Update button appearance
    const ttsButton = document.getElementById('tts-toggle');
    if (ttsButton) {
        ttsButton.innerHTML = ttsEnabled ? '🔊 TTS On' : '🔈 TTS Off';
        ttsButton.classList.toggle('active', ttsEnabled);
    }
    
    // Show all or hide all speak buttons based on state
    const speakButtons = document.querySelectorAll('.speak-button');
    speakButtons.forEach(btn => {
        btn.style.display = ttsEnabled ? 'block' : 'none';
    });
}

async function handleSpeak(button, text) {
    try {
        if (button) button.disabled = true;
        console.log('Sending text to TTS:', text);

        const response = await fetch('/text-to-speech', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text })
        });
        
        const data = await response.json();
        if (data.audio) {
            console.log('Received audio data, attempting to play...');
            const audio = new Audio('data:audio/mp3;base64,' + data.audio);
            
            // Create a silent audio context to unlock audio on first interaction
            if (!window.audioContextInitialized) {
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                await audioContext.resume();
                window.audioContextInitialized = true;
            }
            
            try {
                await audio.play();
            } catch (playError) {
                console.error('Playback error:', playError);
                // Retry once after a short delay
                setTimeout(async () => {
                    try {
                        await audio.play();
                    } catch (retryError) {
                        console.error('Retry playback failed:', retryError);
                    }
                }, 100);
            }
        } else {
            console.error('No audio data received');
        }
    } catch (error) {
        console.error('TTS Error:', error);
    } finally {
        if (button) button.disabled = false;
    }
}