"""Hosted entry point for the FRC plate pocketing UI."""

import os

from gradio import networking

from frcpocket.app import buildApp

# Gradio pings itself at http://localhost:<port>/ after starting and refuses to run if
# that fails. Inside Cloud Run's sandbox the ping fails even though the app serves fine
# on 0.0.0.0, and Cloud Run health-checks the port itself, so skip the ping there.
if os.environ.get("K_SERVICE"):  # set by Cloud Run
    networking.url_ok = lambda url: True

# Only the main process starts the site: the FEA worker processes are spawned, and a
# spawned process re-imports this file. No host or port here: the host sets them through
# GRADIO_SERVER_NAME / GRADIO_SERVER_PORT.
if __name__ == "__main__":
    buildApp().launch()
