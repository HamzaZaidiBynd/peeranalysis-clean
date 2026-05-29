from __future__ import annotations

import os

from api.index import app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8766"))
    app.run(host="127.0.0.1", port=port, debug=False)
