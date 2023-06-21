from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

from pathlib import Path

def main():
    # token_path = r'/mnt/beegfs/homes/fquattrini/client_secret_1080745903255-gvde47k5b0g4k7gp5fj8qm946j68uqej.apps.googleusercontent.com.json'
    # creds = Credentials.from_authorized_user_file(token_path, ['https://www.googleapis.com/auth/drive'])
    # drive_service = build('drive', 'v3', credentials=creds)

    google_auth = GoogleAuth()
    google_auth.LocalWebserverAuth()
    print(f'Done!')
    drive = GoogleDrive(google_auth)
    

if __name__ == '__main__':
    main()