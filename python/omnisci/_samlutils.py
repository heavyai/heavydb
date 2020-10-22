import re
import requests
from html import unescape
from urllib.parse import urlparse


def get_saml_response(
    idpurl,
    username,
    password,
    userformfield,
    passwordformfield,
    sslverify=True,
):
    """
    Obtains the SAML response from an Identity Provider
    given the provided username and password.

    Parameters
    ----------
    idpurl : str
        The logon page of the SAML Identity Provider
    username : str
        SAML Username
    password : str
        SAML Password
    userformfield : str
        The HTML form ID for the username
    passwordformfield : str
        The HTML form ID for the password
    sslverify : bool, optional
        Verify TLS certificates, by default True
    """

    session = requests.Session()

    response = session.get(idpurl, verify=sslverify)
    initialurl = response.url
    formaction = initialurl
    # print(page.content)

    # Determine if there's an action in the form, if there is,
    # use it instead of the page URL
    asearch = re.search(
        r'<form\s+.*?\s+action' r'\s*=\s*\"(.*?)\".*?<\s*/form>',
        response.text,
        re.IGNORECASE | re.DOTALL,
    )

    if asearch:
        formaction = asearch.group(1)

    # If the action is a path not a URL, build the full
    if not formaction.lower().startswith('http'):
        parsedurl = urlparse(idpurl)
        formaction = parsedurl.scheme + "://" + parsedurl.netloc + formaction

    # Un-urlencode the URL
    formaction = unescape(formaction)

    formpayload = {userformfield: username, passwordformfield: password}

    response = session.post(formaction, data=formpayload, verify=sslverify)

    samlresponse = None
    ssearch = re.search(
        r'<input\s+.*?\s+name\s*=\s*'
        r'\"SAMLResponse\".*?\s+value=\"(.*?)\".*?\/>',
        response.text,
        re.IGNORECASE | re.DOTALL,
    )
    if ssearch:
        samlresponse = ssearch.group(1)
        # Remove any whitespace, some providers include
        # new lines in the response (!)
        re.sub(r"[\r\n\t\s]*", "", samlresponse)

    if not samlresponse:
        raise ValueError('No SAMLResponse found in response.')

    return samlresponse
