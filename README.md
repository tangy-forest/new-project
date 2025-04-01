# Pharmacokinetic Overlap Analysis

A web application for analyzing and visualizing drug concentration overlaps between different compounds.

## Deployment on Vercel

1. Install the Vercel CLI:
```bash
npm install -g vercel
```

2. Login to Vercel:
```bash
vercel login
```

3. Deploy the application:
```bash
vercel
```

4. To share with collaborators:
   - Go to your Vercel dashboard
   - Select the project
   - Click on "Settings" > "Collaboration"
   - Add collaborator email addresses

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run src/app.py
```

## Environment Variables

No environment variables are required for basic functionality.

## Features

- Upload compound data via CSV
- Interactive concentration-time plots
- Normalized concentration comparison
- Overlap score calculation
- Image-based curve extraction
- LaTeX rendered pharmacokinetic formulas

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License