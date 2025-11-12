// server.js - Optimized and updated with correct LangChain imports

const express = require('express');
const path = require('path');
// --- CORRECTED IMPORTS ---
const { HNSWLib } = require("@langchain/community/vectorstores/hnswlib");
const { GoogleGenerativeAIEmbeddings } = require("@langchain/google-genai");
// --- END CORRECTED IMPORTS ---
const { callBridgeBuddy } = require('./rag-tutor.js');

// --- CONFIGURATION ---
const VECTOR_STORE_PATH = path.join(__dirname, 'vector_store');
const API_KEY = "AIzaSyAk49CAU0bTTDWzAEyZNt4ZMj3JeT1e1Kk";
// --------------------

const app = express();
const port = 3000;

let vectorStore; 

async function initializeVectorStore() {
  try {
    console.log('🚀 Initializing knowledge base...');
    const embeddings = new GoogleGenerativeAIEmbeddings({ apiKey: API_KEY });
    vectorStore = await HNSWLib.load(VECTOR_STORE_PATH, embeddings);
    console.log('✅ Knowledge base loaded and ready.');
  } catch (error) {
    console.error('❌ Failed to initialize vector store:', error);
    process.exit(1);
  }
}

app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());

app.post('/ask-buddy', async (req, res) => {
  try {
    const userPrompt = req.body.prompt;
    if (!userPrompt) {
      return res.status(400).json({ error: 'Prompt is required' });
    }
    console.log(`Received prompt: "${userPrompt}"`);

    if (!vectorStore) {
        throw new Error("Vector store is not initialized.");
    }

    const results = await vectorStore.similaritySearch(userPrompt, 1);
    const context = results.map(doc => doc.pageContent).join('\n\n---\n\n');
    console.log(`\n📚 Found relevant context:\n${context}\n`);

    const buddyResponse = await callBridgeBuddy(userPrompt, context);
    res.json({ response: buddyResponse });

  } catch (error) {
    console.error('Error in /ask-buddy endpoint:', error);
    res.status(500).json({ error: 'Failed to get response from Bridge Buddy' });
  }
});

app.listen(port, () => {
  console.log(`✅ RAG-powered Bridge Buddy server is running on http://localhost:${port}`);
  initializeVectorStore();
});