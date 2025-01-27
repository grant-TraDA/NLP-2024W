async function analyzeContent() {
  // Get the learning topic from storage
  const data = await chrome.storage.local.get('learningTopic');
  const topic = data.learningTopic || '';

  // Get the page content
  const pageContent = document.body.innerText;

  // Prepare the data to match your FastAPI model
  const requestData = {
    topic: topic,
    content: pageContent
  };

  try {
    const response = await fetch('http://127.0.0.1:8001/validate-webpage', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestData)
    });

    const result = await response.json();
    
    // If result is false, block the page
    if (!result.result) {
      // Store the previous URL before blocking
      const previousUrl = document.referrer;
      
      // Create blocking overlay
      const overlay = document.createElement('div');
      overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.9);
        color: white;
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 10000;
      `;
      
      overlay.innerHTML = `
        <div style="text-align: center;">
          <h1>Content Blocked</h1>
          <p>This content doesn't match your learning goals.</p>
          <button id="goBackButton" style="padding: 10px 20px; margin: 5px;">Go Back</button>
          <button id="goHomeButton" style="padding: 10px 20px; margin: 5px;">Go to Google</button>
        </div>
      `;
      
      document.body.appendChild(overlay);

      // Add event listener for the Go Back button
      document.getElementById('goBackButton').addEventListener('click', () => {
        if (previousUrl) {
          window.location.href = previousUrl;
        } else {
          window.history.back();
        }
      });

      // Add event listener for the Home button
      document.getElementById('goHomeButton').addEventListener('click', () => {
        window.location.href = 'https://www.google.com';
      });
    }
  } catch (error) {
    console.error('Error analyzing content:', error);
  }
}

// Run analysis when page loads
analyzeContent();