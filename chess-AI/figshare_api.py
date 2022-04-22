"""Module for retrieving and uploading articles through the Figshare API."""
import hashlib
import json
import os

from urllib.request import urlretrieve
import requests
from requests.exceptions import HTTPError

class FigshareApi:
    """Class with all the functions for interacting with Figshare."""
    # Size of upload chunks
    CHUNK_SIZE = 1048576
    # Figshare base URL
    BASE_URL = "https://api.figshare.com/v2/{endpoint}"
    # Get key from environment variale
    API_KEY = os.getenv("FIGSHARE_KEY")

    @staticmethod
    def get_figshare_article(store_dir, file_name):
        """Get an article from figshare and store it in store_path.

        file_name refers to the figshare filename, not title.
        """
        endpoint = "account/articles/{article_id}/files"
        download_url = None
        store_path = store_dir + file_name
        # Get all the articles within figshare
        articles = FigshareApi.get_articles()
        # Iterate through all the returned articles
        for article in articles:
            article_id = article["id"]
            file_data = FigshareApi.issue_request("GET", endpoint.format(article_id=article_id))
            # Ensure articles has associated file data
            if len(file_data) > 0:
                # Get the file name of the article
                figshare_file = file_data[0]["name"]
                # See if this the file being searched for
                if figshare_file == file_name:
                    # Get the download URL
                    download_url = file_data[0]["download_url"]
                    break
        if not download_url:
            print("File %s not found in Figshare" % (file_name))
            return False

        urlretrieve(download_url, store_path)
        print("File %s downloaded to %s" % (file_name, store_path))
        # Return true to signify file was found
        return True

    @staticmethod
    def raw_issue_request(method, url, data=None, binary=False):
        """Utility function taken from Figshare documentation."""
        headers = {'Authorization': 'token ' + FigshareApi.API_KEY}

        if data is not None and not binary:
            data = json.dumps(data)

        response = requests.request(method, url, headers=headers, data=data)
        try:
            response.raise_for_status()
            try:
                data = json.loads(response.content)
            except ValueError:
                data = response.content
        except HTTPError as error:
            print("Caught an HTTPError: {}".format(error.message))
            print('Body:\n', response.content)
            raise

        return data

    @staticmethod
    def issue_request(method, endpoint, *args, **kwargs):
        """Utility function taken from Figshare documentation."""
        return FigshareApi.raw_issue_request(method, FigshareApi.BASE_URL.format(endpoint=endpoint), \
        *args, **kwargs)

    @staticmethod
    def create_article(title, desc, keywords, categories):
        """Create a figshare article given the title, description, keywords
        and categories.

        These are all fields that are required for an article to be published
        in Figshare.
        """
        data = {
            'title': title,
            "description": desc,
            "keywords": keywords,
            "categories": categories
        }
        result = FigshareApi.issue_request('POST', 'account/articles', data=data)
        print('Created article:', result['location'], '\n')

        result = FigshareApi.raw_issue_request('GET', result['location'])
        return result['id']

    @staticmethod
    def get_file_check_data(file_name):
        """Utility function taken from Figshare documentation."""
        with open(file_name, 'rb') as fin:
            md5 = hashlib.md5()
            size = 0
            data = fin.read(FigshareApi.CHUNK_SIZE)
            while data:
                size += len(data)
                md5.update(data)
                data = fin.read(FigshareApi.CHUNK_SIZE)

            return md5.hexdigest(), size

    @staticmethod
    def initiate_new_upload(article_id, file_name):
        """Utility function taken from Figshare documentation."""
        endpoint = 'account/articles/{}/files'
        endpoint = endpoint.format(article_id)

        md5, size = FigshareApi.get_file_check_data(file_name)
        data = {'name': os.path.basename(file_name),
                'md5': md5,
                'size': size,
                }

        result = FigshareApi.issue_request('POST', endpoint, data=data)
        print('Initiated file upload:', result['location'], '\n')

        result = FigshareApi.raw_issue_request('GET', result['location'])

        return result

    @staticmethod
    def get_articles():
        """Get all the articles in Figshare."""
        result = FigshareApi.issue_request('GET', 'account/articles')
        return result

    @staticmethod
    def complete_upload(article_id, file_id):
        """Utility function taken from Figshare documentation."""
        FigshareApi.issue_request('POST', 'account/articles/{}/files/{}'.format(article_id, file_id))

    @staticmethod
    def upload_parts(file_info, path):
        """Utility function taken from Figshare documentation."""
        url = '{upload_url}'.format(**file_info)
        result = FigshareApi.raw_issue_request('GET', url)

        print('Uploading parts:')
        with open(path, 'rb') as fin:
            for part in result['parts']:
                FigshareApi.upload_part(file_info, fin, part)

    @staticmethod
    def upload_part(file_info, stream, part):
        """Utility function taken from Figshare documentation."""
        udata = file_info.copy()
        udata.update(part)
        url = '{upload_url}/{partNo}'.format(**udata)

        stream.seek(part['startOffset'])
        data = stream.read(part['endOffset'] - part['startOffset'] + 1)

        FigshareApi.raw_issue_request('PUT', url, data=data, binary=True)
        print('  Uploaded part {partNo} from {startOffset} to {endOffset}'.format(**part))

    @staticmethod
    def upload(title, desc, keywords, categories, path):
        """Upload and publish an article to figshare.

        Articles do not become accessible through the API until they are
        published.
        """
        # Create the article and get associated ID
        article_id = FigshareApi.create_article(title, desc, keywords, categories)
        # Upload the file to Figshare
        file_info = FigshareApi.initiate_new_upload(article_id, path)
        FigshareApi.upload_parts(file_info, path)
        FigshareApi.complete_upload(article_id, file_info['id'])
        # The article must be published
        FigshareApi.issue_request('POST', f"account/articles/{article_id}/publish".format(article_id))
