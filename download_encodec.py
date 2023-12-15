import requests



def _get_checkpoint_url(root_url: str, checkpoint: str):
    if not root_url.endswith('/'):
        root_url += '/'
    return root_url + checkpoint


if __name__ == '__main__':
    ROOT_URL = 'https://dl.fbaipublicfiles.com/encodec/v0/'
    
    # download 24k model
    checkpoint_name = 'encodec_24khz-d7cc33bc.th'
    url = _get_checkpoint_url(ROOT_URL, checkpoint_name)
    dest_path = 'pretrained/{}'.format(checkpoint_name)
    response = requests.get(url)
    if response.status_code == 200:
        with open(dest_path, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded to {dest_path}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")

    # download 48k model
    checkpoint_name = 'encodec_48khz-7e698e3e.th'
    url = _get_checkpoint_url(ROOT_URL, checkpoint_name)
    dest_path = 'pretrained/{}'.format(checkpoint_name)
    response = requests.get(url)
    if response.status_code == 200:
        with open(dest_path, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded to {dest_path}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")