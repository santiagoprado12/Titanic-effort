import subprocess
import os
import json
from cryptography.fernet import Fernet



class KeysExtraction:
    """
    This class is used to decrypt the aws keys from the encrypted_keys.txt file.
    """

    def __init__(self, path:str = "encrypted_keys.txt"):

        self.get_secret_key()
        self.path = path

    def get_secret_key(self):

        VAR_NAME = "SECRET_KEY"     
        #see if env variable exists
        if  VAR_NAME in os.environ:
            self._secret_key = os.environ[VAR_NAME]
        else:
            self._secret_key = input("Enter the secret key: ")

    def decrypt_file(self) -> dict:

        with open(self.path, "rb") as file:
            encrypted_data = file.read()

        cipher_suite = Fernet(str.encode(self._secret_key))
        decrypted_data = cipher_suite.decrypt(encrypted_data)
        self.decrypted_json_data = json.loads(decrypted_data.decode())

        return self.decrypted_json_data
    
    def set_env_variables(self):
        
        keys = self.decrypt_file()
        for key, value in keys.items():
            os.environ[key] = value
    


