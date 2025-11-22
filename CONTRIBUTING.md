# Contributing to TSU

Thank you for your interest in contributing to TSU!

## Development Setup

```bash
git clone https://github.com/Arsham-001/tsu-emulator
cd tsu-emulator
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running Tests

```bash
pytest tests/ -v
```

All 121 tests should pass.

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Add tests if applicable
5. Ensure all tests pass
6. Commit your changes (`git commit -m 'Add feature'`)
7. Push to your branch (`git push origin feature/your-feature`)
8. Open a Pull Request

## Code Style

- Follow existing code patterns
- Add docstrings to new public functions
- Keep comments informative and professional
- Run tests before submitting

## Reporting Issues

- Check existing issues first
- Provide minimal reproducible example
- Include Python version and OS
- Include error messages if applicable

## Questions?

Open an issue or email arsham.rocky21@gmail.com
