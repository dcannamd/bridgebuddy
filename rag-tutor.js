// rag-tutor.js - The "Brain" upgraded for RAG

const { GoogleGenerativeAI } = require("@google/generative-ai");

// --- CONFIGURATION ---
const API_KEY = "AIzaSyAk49CAU0bTTDWzAEyZNt4ZMj3JeT1e1Kk";
// --------------------

const genAI = new GoogleGenerativeAI(API_KEY);

// The main function now accepts 'context' from our knowledge base
async function callBridgeBuddy(prompt, context) {
  try {
    const model = genAI.getGenerativeModel({
      model: "gemini-2.5-flash-preview-05-20",
      // The system instruction is now updated to prioritize the provided context
      systemInstruction: {
        parts: [{
          text: `You are an expert instructional designer and senior software engineer specializing in ProtoPie, ProtoPie Bridge, NodeJS, and JSON. Your name is "Bridge Buddy."
          Your purpose is to act as a friendly, encouraging, and patient tutor.

          *** IMPORTANT RULE ***
          You MUST use the provided "CONTEXT" section below to answer the user's question. The context contains specific, approved code examples and documentation from the user's private knowledge base.
          Prioritize the information in the context above your own general knowledge. If the context provides a direct answer, use it. If not, use the context to inform your response.

          Your other rules are:
          1. Always provide clear, step-by-step explanations.
          2. When providing code from the context, always explain it piece by piece.
          3. Maintain a positive and supportive tone. Use emojis where appropriate.`
        }]
      }
    });

    // We no longer need a static history. The dynamic context is more powerful.
    const chat = model.startChat();

    // We construct a new prompt that includes the context we found
    const augmentedPrompt = `
      CONTEXT:
      ---
      ${context}
      ---

      Based on the context above, please answer the following question:
      QUESTION: "${prompt}"
    `;

    const result = await chat.sendMessage(augmentedPrompt);
    const response = result.response;
    const text = response.text();

    return text;

  } catch (error) {
    console.error("An error occurred in rag-tutor:", error);
    throw new Error("Failed to call the AI model");
  }
}

module.exports = { callBridgeBuddy };