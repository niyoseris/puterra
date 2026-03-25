// Patch to add web search keywords and session persistence
// Keywords that should trigger web search directly
const needsWebSearch = /hava|weather|haber|news|güncel|current|bugün|today|bu hafta|this week|bu ay|this month|sinema|cinema|film|movie|vizyon|fiyat|price|kaç|ne kadar|ne zaman|when|kim|who|nerede|where|son|latest|yeni|new|2024|2025|2026|son dakika|günün|şu an|neler|hangisi|hangi|kktc|kıbrıs|lefkosa|girne|mağusa/i.test(message);

// Clear chat command check
if (/^(?:clear|temizle|new chat|yeni sohbet)$/i.test(message)) {
    clearChatHistory();
    document.getElementById(loadingId)?.remove();
    return;
}

// Save after each message
setTimeout(saveChatHistory, 100);