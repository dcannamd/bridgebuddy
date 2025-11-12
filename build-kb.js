// build-kb.js - Final, stable version with correct batch processing

const path = require('path');
const { HNSWLib } = require("@langchain/community/vectorstores/hnswlib");
const { GoogleGenerativeAIEmbeddings } = require("@langchain/google-genai");
const { DirectoryLoader } = require("langchain/document_loaders/fs/directory");
const { TextLoader } = require("langchain/document_loaders/fs/text");
const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter");

// --- CONFIGURATION ---
const KNOWLEDGE_BASE_DIR = path.join(__dirname, 'knowledge-base');
const VECTOR_STORE_PATH = path.join(__dirname, 'vector_store');
const API_KEY = "AIzaSyAk49CAU0bTTDWzAEyZNt4ZMj3JeT1e1Kk"; // Make sure this is set
const BATCH_SIZE = 100; // Process chunks in batches
// --------------------

// Helper function to add a small delay between API calls
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

async function main() {
  console.log('🚀 Starting knowledge base build with robust batch processing...');

  try {
    // 1. Load all supported documents
    console.log('Loading documents...');
    const loader = new DirectoryLoader(KNOWLEDGE_BASE_DIR, {
      ".md": (path) => new TextLoader(path),
      ".js": (path) => new TextLoader(path),
      ".json": (path) => new TextLoader(path),
      ".css": (path) => new TextLoader(path),
      ".html": (path) => new TextLoader(path),
    });
    const docs = await loader.load();
    console.log(`Loaded ${docs.length} documents.`);

    // 2. Split documents and filter out any empty ones
    const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 800, chunkOverlap: 80 });
    const splitDocs = await splitter.splitDocuments(docs);
    const nonEmptyDocs = splitDocs.filter(doc => doc.pageContent.trim() !== "");
    console.log(`Found ${nonEmptyDocs.length} non-empty chunks to process.`);

    if (nonEmptyDocs.length === 0) {
      console.log('⚠️ No content to process. Halting build.');
      return;
    }

    // 3. Create embeddings model
    console.log('Creating embeddings model...');
    const embeddings = new GoogleGenerativeAIEmbeddings({ apiKey: API_KEY });

    // --- NEW BATCH PROCESSING WORKFLOW ---
    // 4. Initialize the vector store with the FIRST document
    console.log('Initializing vector store with the first document...');
    const vectorStore = await HNSWLib.fromDocuments([nonEmptyDocs[0]], embeddings);
    console.log('✅ Vector store initialized successfully.');

    // 5. Process the REST of the documents in batches
    const remainingDocs = nonEmptyDocs.slice(1);
    const totalBatches = Math.ceil(remainingDocs.length / BATCH_SIZE);
    console.log(`Starting to process ${remainingDocs.length} remaining chunks in ${totalBatches} batches of ${BATCH_SIZE}...`);

    for (let i = 0; i < remainingDocs.length; i += BATCH_SIZE) {
      const batch = remainingDocs.slice(i, i + BATCH_SIZE);
      if (batch.length === 0) continue;

      const currentBatchNumber = Math.floor(i / BATCH_SIZE) + 1;
      console.log(`Processing batch ${currentBatchNumber}/${totalBatches}...`);
      
      try {
        // Use the correct 'addDocuments' instance method
        await vectorStore.addDocuments(batch);
      } catch (e) {
        // This will catch errors within a batch (e.g., from empty embeddings) and skip that batch
        console.warn(`⚠️ An error occurred in batch ${currentBatchNumber}. Some documents may have been skipped. Error: ${e.message}`);
      }
      
      // Add a small delay to be kind to the API rate limits
      await sleep(200);
    }
    // --- END BATCH PROCESSING ---

    console.log('Vector store built successfully in memory.');

    // 6. Save the fully populated vector store
    console.log(`Saving vector store to: ${VECTOR_STORE_PATH}...`);
    await vectorStore.save(VECTOR_STORE_PATH);
    console.log(`✅ Vector store saved successfully.`);

    console.log('\n🎉 Knowledge base build complete!');

  } catch (error) {
    console.error('❌ An error occurred during the build process:', error);
  }
}

main();
