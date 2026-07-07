from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as vuetify

server = get_server()

with SinglePageLayout(server) as layout:
    layout.title.set_text("Test App")
    with layout.content:
        vuetify.VContainer("Hello Trame")

if __name__ == "__main__":
    server.start()
