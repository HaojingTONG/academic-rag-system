# Academic RAG System - Frontend Dashboard

A modern, lightweight web interface for the Academic RAG (Retrieval-Augmented Generation) System.

## Features

### 🔍 Ask Panel
- **Query Input**: Natural language question input
- **Parameter Control**: Adjust `top_k` for number of retrieved sources
- **Answer Display**: Generated answers with citation markers
- **Source Visualization**: View retrieved document chunks with relevance scores
- **Debug Mode**: Toggle raw JSON response view

### 📥 Ingest Panel
- **File Upload**: Upload PDF or TXT academic papers
- **Index Management**: Refresh vector database to include new papers
- **Metrics Display**: View indexing statistics (retrieved, kept, latency)
- **File Validation**: Automatic file type checking

### 🔧 Debug Panel
- **Health Check**: Monitor RAG system status
- **Performance Metrics**: View retrieval and latency statistics
- **Trace ID Support**: Track requests with trace identifiers
- **History**: Keep track of recent health checks
- **System Info**: Display configuration and API details

## Tech Stack

- **Framework**: React 18 + TypeScript
- **Build Tool**: Vite 5
- **Styling**: Tailwind CSS 3
- **HTTP Client**: Fetch API

## Prerequisites

- Node.js 18+ or Bun
- npm/yarn/pnpm/bun
- Running RAG backend server (default: http://localhost:8000)

## Quick Start

### 1. Installation

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies (choose one)
npm install
# or
yarn install
# or
pnpm install
# or
bun install
```

### 2. Configuration

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` to configure your backend URL:

```env
VITE_RAG_BASE_URL=http://localhost:8000
```

### 3. Development

```bash
# Start development server
npm run dev
# or
yarn dev
# or
bun dev

# The app will open at http://localhost:3000
```

### 4. Production Build

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── AskPanel.tsx       # Query interface
│   │   ├── IngestPanel.tsx    # Upload & indexing
│   │   └── DebugPanel.tsx     # Health check & debug
│   ├── RAGDashboard.tsx       # Main dashboard component
│   ├── App.tsx                # Root app component
│   ├── main.tsx               # Entry point
│   └── index.css              # Global styles
├── public/                     # Static assets
├── .env.example               # Environment variables template
├── index.html                 # HTML template
├── package.json               # Dependencies
├── vite.config.ts             # Vite configuration
├── tailwind.config.js         # Tailwind configuration
├── tsconfig.json              # TypeScript configuration
└── README.md                  # This file
```

## API Endpoints

The frontend expects the following backend endpoints:

### Query
```http
POST /rag/query
Content-Type: application/json

{
  "query": "What is the Transformer architecture?",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": {
    "text": "The Transformer is...",
    "citations": [1, 2, 3]
  },
  "retrieved": [
    {
      "content": "...",
      "score": 0.95,
      "metadata": {}
    }
  ],
  "trace_id": "abc123",
  "success": true
}
```

### Upload Paper
```http
POST /papers/upload
Content-Type: multipart/form-data

file: <PDF or TXT file>
```

### Refresh Index
```http
POST /index/refresh
Content-Type: application/json
```

**Response:**
```json
{
  "retrieved_n": 10,
  "kept_n": 8,
  "latency_ms": 150,
  "message": "Index refreshed successfully"
}
```

### Health Check
```http
GET /health/rag
```

**Response:**
```json
{
  "retrieved_n": 10,
  "kept_n": 8,
  "citations": 5,
  "latency_ms": 120,
  "trace_id": "xyz789",
  "status": "healthy"
}
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_RAG_BASE_URL` | Backend API base URL | `http://localhost:8000` |

## Development

### Running with Backend

1. Start the backend server:
```bash
# In the project root
python app/main.py
# or
uvicorn app.main:app --reload --port 8000
```

2. Start the frontend:
```bash
# In the frontend directory
npm run dev
```

3. Open http://localhost:3000 in your browser

### Hot Reload

Vite provides instant hot module replacement (HMR) during development. Changes to React components will be reflected immediately without full page reload.

### Type Checking

```bash
# Run TypeScript type checking
npm run build
```

## Customization

### Styling

The dashboard uses Tailwind CSS for styling. You can customize the theme in `tailwind.config.js`:

```js
export default {
  theme: {
    extend: {
      colors: {
        primary: '#3b82f6',
        // Add custom colors
      }
    }
  }
}
```

### API Base URL

To connect to a different backend:

1. Update `.env`:
```env
VITE_RAG_BASE_URL=https://your-api-domain.com
```

2. Restart the dev server

### Add New Features

1. Create component in `src/components/`
2. Import and use in `RAGDashboard.tsx`
3. Add new tab if needed in the tabs array

## Troubleshooting

### CORS Issues

If you encounter CORS errors:

1. Ensure backend has CORS enabled:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

2. Check `VITE_RAG_BASE_URL` is correct

### Port Already in Use

If port 3000 is in use, change it in `vite.config.ts`:

```ts
export default defineConfig({
  server: {
    port: 3001, // Change to available port
  }
})
```

### Build Errors

1. Clear node_modules and reinstall:
```bash
rm -rf node_modules
npm install
```

2. Clear Vite cache:
```bash
rm -rf node_modules/.vite
```

## Deployment

### Build for Production

```bash
npm run build
```

This creates an optimized build in the `dist/` directory.

### Deploy to Static Hosting

The built `dist/` folder can be deployed to:

- **Vercel**:
```bash
npm install -g vercel
vercel --prod
```

- **Netlify**:
```bash
npm install -g netlify-cli
netlify deploy --prod --dir=dist
```

- **GitHub Pages**, **Cloudflare Pages**, **AWS S3**, etc.

### Environment Variables for Production

Set `VITE_RAG_BASE_URL` to your production backend URL before building:

```bash
# .env.production
VITE_RAG_BASE_URL=https://api.your-domain.com
```

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

## Performance

- **Bundle Size**: ~150KB gzipped
- **First Load**: < 2s
- **Time to Interactive**: < 3s

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT

## Support

For issues or questions:
- Open an issue on GitHub
- Check backend API documentation
- Review browser console for errors

---

**Version**: 2.0.0
**Last Updated**: 2025-01-29
**Maintainer**: Academic RAG Team
