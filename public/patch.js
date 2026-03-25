// Patch to add web search keywords and session persistence
// Keywords that should trigger web search directly
const needsWebSearch = /weather|news|current|today|this week|this month|cinema|movie|price|when|who|where|latest|new|2024|2025|2026|breaking|right now/i.test(message);

// Clear chat command check
if (/^(?:clear|new chat)$/i.test(message)) {
    clearChatHistory();
    document.getElementById(loadingId)?.remove();
    return;
}

// Save after each message
setTimeout(saveChatHistory, 100);