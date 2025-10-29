# Quick Start Guide

Get the RAG Dashboard running in 3 minutes! ⚡

## Prerequisites

- Node.js 18+ installed
- Backend server running on `http://localhost:8000`

## Installation & Setup

```bash
# 1. Navigate to frontend directory
cd frontend

# 2. Install dependencies
npm install

# 3. Configure environment (optional)
cp .env.example .env
# Edit .env if your backend runs on a different URL

# 4. Start development server
npm run dev
```

The dashboard will automatically open at **http://localhost:3000** 🎉

## First Steps

### 1. Test the Connection

Go to the **Debug** tab and click "Check Health" to verify backend connectivity.

### 2. Upload a Paper

Switch to the **Ingest** tab:
1. Click the upload area
2. Select a PDF or TXT file
3. Click "Upload Paper"
4. Click "Refresh Index" to make it searchable

### 3. Ask Questions

Go to the **Ask** tab:
1. Enter your question (e.g., "What is the Transformer architecture?")
2. Click "Ask"
3. View the generated answer with citations
4. Scroll down to see retrieved source documents

## Troubleshooting

### Backend Connection Failed

Check that your backend is running:

```bash
# In the main project directory
python app/main.py
# or
uvicorn app.main:app --reload --port 8000
```

### Port 3000 Already in Use

Edit `vite.config.ts` and change the port:

```ts
export default defineConfig({
  server: {
    port: 3001, // Change to any available port
  }
})
```

### CORS Errors

Ensure your backend has CORS middleware enabled for `http://localhost:3000`.

## What's Next?

- Read the full [README.md](./README.md) for detailed documentation
- Customize the theme in `tailwind.config.js`
- Add new features in `src/components/`

## Support

- Check browser console for errors
- Verify backend API is responding
- Review `.env` configuration

---

**Happy querying!** 🚀📚
