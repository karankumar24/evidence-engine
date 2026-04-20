"""Local filesystem storage for uploaded files, organised by packet ID."""

import os
import shutil
from pathlib import Path


class FileStore:
    """Manages file storage on the local filesystem.

    Files are stored at: {base_dir}/{packet_id}/{filename}
    """

    def __init__(self, base_dir: str) -> None:
        self.base_dir = str(Path(base_dir).resolve())
        os.makedirs(self.base_dir, exist_ok=True)

    def save(self, packet_id: str, filename: str, content: bytes) -> str:
        """Save file content to disk and return the full file path.

        Defense-in-depth against path traversal and symlink attacks: after
        joining + resolving the target path, we assert it still sits under
        the resolved packet directory.

        Args:
            packet_id: UUID string for the packet (used as subdirectory).
            filename: Original filename (basename is taken; any directory
                components are stripped before joining).
            content: Raw file bytes.

        Returns:
            Absolute file path where the file was saved.

        Raises:
            ValueError: If the resolved path escapes packet_dir.
        """
        packet_dir = (Path(self.base_dir) / str(packet_id)).resolve()
        packet_dir.mkdir(parents=True, exist_ok=True)
        safe_name = os.path.basename(filename) or "upload.bin"
        file_path = (packet_dir / safe_name).resolve()
        # Must still be inside packet_dir after resolve (catches symlink escape)
        if not str(file_path).startswith(str(packet_dir) + os.sep):
            raise ValueError(
                f"Refused to write outside packet directory: {filename!r}"
            )
        with open(file_path, "wb") as f:
            f.write(content)
        return str(file_path)

    def delete_packet(self, packet_id: str) -> None:
        """Remove the entire packet directory and all files within it.

        Args:
            packet_id: UUID string for the packet.
        """
        packet_dir = os.path.join(self.base_dir, str(packet_id))
        if os.path.exists(packet_dir):
            shutil.rmtree(packet_dir)
