import streamlit as st
import streamlit.components.v1 as components


def inject_ga():
    """
    Inject Google Analytics tracking code.
    Note: While GA documentation recommends placing the code in <head>,
    Streamlit doesn't provide direct access to <head>. The tracking
    will still work when injected into the body.
    """
    GA_JS = """
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-Q0DP7DD84W"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());

        gtag('config', 'G-Q0DP7DD84W');
    </script>
    """
    components.html(GA_JS, height=0)
