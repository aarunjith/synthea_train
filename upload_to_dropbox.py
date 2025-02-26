#!/usr/bin/env python3
import os
import sys
import dropbox
from dropbox.files import WriteMode
from dropbox.exceptions import ApiError, AuthError

def upload_file(file_path, dropbox_path, token):
    """Upload a file to Dropbox using the provided access token."""
    dbx = dropbox.Dropbox(token)
    
    # Check that the access token is valid
    try:
        dbx.users_get_current_account()
    except AuthError:
        sys.exit("ERROR: Invalid access token; try re-generating an "
                 "access token from the app console on the web.")
    
    file_size = os.path.getsize(file_path)
    
    print(f"Uploading {file_path} to Dropbox as {dropbox_path} ({file_size / (1024 * 1024):.2f} MB)")
    
    with open(file_path, 'rb') as f:
        # Use chunked upload for large files
        if file_size <= 150 * 1024 * 1024:  # 150 MB max for regular upload
            try:
                dbx.files_upload(
                    f.read(),
                    dropbox_path,
                    mode=WriteMode('overwrite')
                )
                print(f"Uploaded {file_path} to {dropbox_path}")
            except ApiError as err:
                sys.exit(f"API error: {err}")
        else:
            # Chunked upload for files larger than 150 MB
            chunk_size = 4 * 1024 * 1024  # 4 MB per chunk
            
            upload_session_start_result = dbx.files_upload_session_start(f.read(chunk_size))
            cursor = dropbox.files.UploadSessionCursor(
                session_id=upload_session_start_result.session_id,
                offset=f.tell()
            )
            commit = dropbox.files.CommitInfo(path=dropbox_path, mode=WriteMode('overwrite'))
            
            uploaded = f.tell()
            print(f"Uploaded {uploaded / (1024 * 1024):.2f} MB")
            
            while f.tell() < file_size:
                if (file_size - f.tell()) <= chunk_size:
                    dbx.files_upload_session_finish(
                        f.read(chunk_size),
                        cursor,
                        commit
                    )
                else:
                    dbx.files_upload_session_append_v2(
                        f.read(chunk_size),
                        cursor
                    )
                    cursor.offset = f.tell()
                
                uploaded = f.tell()
                print(f"Uploaded {uploaded / (1024 * 1024):.2f} MB of {file_size / (1024 * 1024):.2f} MB")
            
            print(f"Finished uploading {file_path} to {dropbox_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python upload_to_dropbox.py <file_path> <access_token>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    access_token = sys.argv[2]
    
    # Extract filename from path
    file_name = os.path.basename(file_path)
    dropbox_path = f"/{file_name}"
    
    upload_file(file_path, dropbox_path, access_token) 