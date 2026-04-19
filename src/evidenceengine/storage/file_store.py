"""Local filesystem storage for uploaded files, organised by packet ID."""

import os
import shutil


class FileStore:
    """Manages file storage on the local filesystem.

    Files are stored at: {base_dir}/{packet_id}/{filename}
    """

    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def save(self, packet_id: str, filename: str, content: bytes) -> str:
        """Save file content to disk and return the full file path.

        Args:
            packet_id: UUID string for the packet (used as subdirectory).
            filename: Original filename.
            content: Raw file bytes.

        Returns:
            Absolute file path where the file was saved.
        """
        packet_dir = os.path.join(self.base_dir, str(packet_id))
        os.makedirs(packet_dir, exist_ok=True)
        file_path = os.path.join(packet_dir, os.path.basename(filename))
        with open(file_path, "wb") as f:
            f.write(content)
        return file_path

    def delete_packet(self, packet_id: str) -> None:
        """Remove the entire packet directory and all files within it.

        Args:
            packet_id: UUID string for the packet.
        """
        packet_dir = os.path.join(self.base_dir, str(packet_id))
        if os.path.exists(packet_dir):
            shutil.rmtree(packet_dir)
