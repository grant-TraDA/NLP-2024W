document.getElementById('saveTopic').addEventListener('click', () => {
    const topic = document.getElementById('learningTopic').value;
    chrome.storage.local.set({ learningTopic: topic }, () => {
      console.log('Topic saved:', topic);
    });
  });