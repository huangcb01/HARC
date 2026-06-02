import os
import sys
import shutil
import psutil
import subprocess
import signal
import time
import asyncio
import logging
import threading
import functools
from pathlib import Path
from typing import Iterable, Optional

from rich.console import Console
from rich.live import Live
from rich.text import Text
from utils import get_logger

logger = get_logger(__name__)


class RichStatusUI:
    def __init__(self):
        self.enabled = bool(sys.stderr.isatty())
        self.console: Optional[Console] = None
        self._live: Optional[Live] = None
        self._status_text = ""
        self._lock = threading.Lock()
        self._saved_root_handlers: list[logging.Handler] = []

    def __enter__(self):
        if not self.enabled:
            return self

        self.console = Console(file=sys.stderr)

        # Replace root logger handlers so all logs play nicely with Live.
        root_logger = get_logger()
        self._saved_root_handlers = list(root_logger.handlers)
        root_logger.handlers = []
        root_logger.propagate = False

        class _ColorConsoleHandler(logging.Handler):
            def __init__(self, console: Console, style: str):
                super().__init__()
                self._console = console
                self._style = style

            def emit(self, record: logging.LogRecord) -> None:
                try:
                    msg = self.format(record)
                    self._console.print(msg, style=self._style)
                except Exception:
                    self.handleError(record)

        handler = _ColorConsoleHandler(self.console, style="red")
        handler.setFormatter(
            logging.Formatter(
                "[%(levelname)s|%(asctime)s] %(filename)s:%(lineno)s >> %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(handler)

        self._live = Live(
            self._renderable(),
            console=self.console,
            refresh_per_second=8,
            transient=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._live is not None:
            try:
                self._live.__exit__(exc_type, exc, tb)
            finally:
                self._live = None

        if self.enabled:
            root_logger = get_logger()
            root_logger.handlers = self._saved_root_handlers
            root_logger.propagate = True
        return False

    def set_status(self, text: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._status_text = text
            if self._live is not None:
                self._live.update(self._renderable())

    def print_child_output(self, text: str) -> None:
        if self.enabled and self.console is not None:
            self.console.print(text, end="", style="white")
        else:
            sys.stderr.write(text)
            sys.stderr.flush()

    def _renderable(self):
        # Allow wrapping so the status bar auto-adjusts line count based on terminal width.
        # Use "fold" so long tokens also wrap instead of being truncated.
        return Text(self._status_text, no_wrap=False, overflow="fold")


class PathItem:
    """A file or directory in the mem_dir.

    We delay syncing until the file/directory is closed and has been stable (no signature change) for `checkpoint_stable_sec`.
    """

    def __init__(self, path: Path, disk_dir: Path):
        self.name = path.name
        self.path = path
        self.disk_path = disk_dir / self.name
        self.disk_tmp_path = disk_dir / f"{self.name}.tmp"
        self.is_dir = path.is_dir()
        self.last_change_ts = 0.0
        self._signature: Optional[tuple[int, int, int]] = None  # (file_count, total_size_bytes, max_mtime_ns)

    def is_stable(self, now: float, stable_sec: float, target_process: psutil.Process) -> bool:
        if now - self.last_change_ts < stable_sec:
            return False
        self._refresh_signature(now)
        return self._is_closed(target_process) and (now - self.last_change_ts) >= stable_sec

    def sync_to_disk(self) -> bool:
        start = time.time()

        if self.disk_tmp_path.exists() or self.disk_tmp_path.is_symlink():
            logger.warning(f"Removing stale tmp checkpoint on disk: {self.disk_tmp_path}")
            _remove_any(self.disk_tmp_path)

        try:
            logger.info(f"Sync checkpoint start: {self.name} -> {self.disk_path}")
            if self.path.is_file() and not self.path.is_symlink():
                self.disk_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.path, self.disk_tmp_path)
                if self.disk_path.exists() or self.disk_path.is_symlink():
                    logger.warning(f"Overwriting existing checkpoint on disk: {self.disk_path}")
                    _remove_any(self.disk_path)
                os.replace(self.disk_tmp_path, self.disk_path)
            else:
                shutil.copytree(
                    self.path,
                    self.disk_tmp_path,
                    symlinks=True,
                    copy_function=shutil.copy2,
                    ignore_dangling_symlinks=True,
                )
                if self.disk_path.exists() or self.disk_path.is_symlink():
                    logger.warning(f"Overwriting existing checkpoint on disk: {self.disk_path}")
                    _remove_any(self.disk_path)
                os.replace(self.disk_tmp_path, self.disk_path)
        except Exception as e:
            logger.error(f"Sync checkpoint {self.name} failed: {e}")
            _remove_any(self.disk_tmp_path)
            return False

        # Free shm and create symlink in mem pointing to disk (only after successful materialize)
        if self.is_dir:
            _remove_any(self.path)
            os.symlink(str(self.disk_path), str(self.path))
        logger.info(f"Sync checkpoint done: {self.name} ({time.time() - start:.1f}s)")
        return True

    def delete_from_disk(self) -> None:
        _remove_any(self.disk_tmp_path)
        _remove_any(self.disk_path)

    def _is_closed(self, target_process: psutil.Process) -> bool:
        pids = [target_process.pid]
        try:
            pids.extend([p.pid for p in target_process.children(recursive=True)])
        except Exception:
            pass

        for pid in pids:
            for target in PathItem._fd_targets_for_pid(pid):
                try:
                    if target == self.path or target.is_relative_to(self.path):
                        return False
                except ValueError:
                    continue
        return True

    def _refresh_signature(self, now: float) -> None:
        file_count, total_size, max_mtime_ns = 0, 0, 0

        try:
            if self.path.is_file() and not self.path.is_symlink():
                st = self.path.stat()
                file_count = 1
                total_size = int(st.st_size)
                max_mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
            else:
                stack = [self.path]
                while stack:
                    d = stack.pop()
                    try:
                        with os.scandir(d) as it:
                            for ent in it:
                                try:
                                    if ent.is_symlink():
                                        continue
                                    if ent.is_dir(follow_symlinks=False):
                                        stack.append(Path(ent.path))
                                    elif ent.is_file(follow_symlinks=False):
                                        st = ent.stat(follow_symlinks=False)
                                        file_count += 1
                                        total_size += int(st.st_size)
                                        mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
                                        if mtime_ns > max_mtime_ns:
                                            max_mtime_ns = mtime_ns
                                except OSError:
                                    continue
                    except OSError:
                        continue
        except Exception as e:
            logger.warning(f"Failed to refresh checkpoint signature: {self.path} for exception {e}")

        sig = (file_count, total_size, max_mtime_ns)
        if self._signature is None or self._signature != sig:
            self._signature = sig
            self.last_change_ts = now

    @staticmethod
    def _fd_targets_for_pid(pid: int) -> Iterable[Path]:
        fd_dir = Path("/proc") / str(pid) / "fd"
        try:
            for fd in fd_dir.iterdir():
                try:
                    yield fd.readlink()
                except OSError:
                    continue
        except OSError:
            return


class TrainingProc:
    def __init__(self, command: str, ui: Optional[RichStatusUI] = None):
        self._ui = ui
        self._forward_threads: list[threading.Thread] = []

        if self._ui is not None and self._ui.enabled:
            self._popen = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self._start_forwarding_threads()
        else:
            self._popen = subprocess.Popen(command, shell=True)
        self.pid = self._popen.pid
        self.process = psutil.Process(self.pid)
        self.paused = False

    def _start_forwarding_threads(self) -> None:
        ui = self._ui
        if ui is None or not ui.enabled:
            return
        if self._popen.stdout is None or self._popen.stderr is None:
            return

        def _pump(stream):
            try:
                for chunk in iter(stream.readline, ""):
                    if chunk == "":
                        break
                    ui.print_child_output(chunk)
            except Exception:
                return

        for s in (self._popen.stdout, self._popen.stderr):
            t = threading.Thread(target=_pump, args=(s,), daemon=True)
            t.start()
            self._forward_threads.append(t)

    def is_running(self) -> bool:
        return self._popen.poll() is None

    def get_return_code(self) -> Optional[int]:
        return self._popen.poll()

    def pause(self):
        children = self.process.children(recursive=True)
        process_list = children + [self.process]
        for p in process_list:
            try:
                p.send_signal(signal.SIGSTOP)
            except psutil.NoSuchProcess:
                logger.warning(f"Process {p.pid} does not exist when stopping.")
            except Exception as e:
                logger.error(f"Failed to stop process {p.pid}: {e}")
        self.paused = True

    def resume(self):
        if not self.paused:
            return
        children = self.process.children(recursive=True)
        process_list = children + [self.process]
        for p in process_list:
            try:
                p.send_signal(signal.SIGCONT)
            except psutil.NoSuchProcess:
                logger.warning(f"Process {p.pid} does not exist when resuming.")
            except Exception as e:
                logger.error(f"Failed to resume process {p.pid}: {e}")
        self.paused = False


def sync(async_func):

    @functools.wraps(async_func)
    def sync_wrapper(*args, **kwargs):
        return asyncio.run(async_func(*args, **kwargs))

    return sync_wrapper


@sync
async def run(
    command: str,
    disk_dir: str,
    mem_dir: str,
    poll_interval_sec: float = 2,
    checkpoint_stable_sec: float = 4,
    backlog_dir_limit: int = 5,
    sync_concurrency: int = 2,
    shm_used_percent_high: float = 90,
    shm_used_percent_low: float = 70,
) -> None:
    ui = RichStatusUI()
    with ui:
        logger.info("Async output writer started.")
        if sync_concurrency < 1:
            raise ValueError(f"sync_concurrency must be >= 1, got {sync_concurrency}")
        disk_path = Path(disk_dir).expanduser().resolve()
        mem_path = Path(mem_dir).expanduser().resolve()
        if mem_path.exists() and not (mem_path.is_dir() and not any(mem_path.iterdir())):
            logger.error(f"mem_dir {mem_path} already exists; please provide an empty or non-existing directory.")
            raise SystemExit(1)
        disk_path.mkdir(parents=True, exist_ok=True)
        mem_path.mkdir(parents=True, exist_ok=True)
        synced_items: list[PathItem] = []
        pending_items: list[PathItem] = []
        inflight: dict[asyncio.Task[bool], PathItem] = {}

        for p in disk_path.iterdir():
            # Clean up stale tmp items on disk (left by previous interrupted copies)
            if p.name.endswith(".tmp"):
                logger.warning(f"Removing stale tmp item on disk: {p.name}")
                _remove_any(p)
            # Create symlinks for existing items on disk so resume works.
            else:
                link_path = mem_path / p.name
                if not link_path.exists():
                    link_path.symlink_to(disk_path / p.name)
                    logger.info(f"Created symlink for existing item on disk: {link_path.name}")
                synced_items.append(PathItem(link_path, disk_path))

        # Start the training process.
        training_proc = TrainingProc(command, ui=ui)

        # Tool functions
        should_pause = (
            lambda n_pending_dirs, shm_used: n_pending_dirs > backlog_dir_limit or shm_used >= shm_used_percent_high
        )
        should_resume = (
            lambda n_pending_dirs, shm_used: n_pending_dirs <= backlog_dir_limit and shm_used <= shm_used_percent_low
        )

        while True:
            training_finished = not training_proc.is_running()

            # Harvest finished sync tasks (if any)
            done_tasks = [t for t in inflight.keys() if t.done()]
            for t in done_tasks:
                item = inflight.pop(t)
                success = False
                try:
                    success = bool(t.result())
                except Exception as e:
                    logger.error(f"Async sync task failed for {item.name}: {e}")
                    success = False

                if success:
                    if item in pending_items:
                        pending_items.remove(item)
                    synced_items.append(item)

            # Scan for newly created file or directory in mem_dir.
            for child in mem_path.iterdir():
                if child.is_symlink() or any(item.name == child.name for item in pending_items + synced_items):
                    continue
                pending_items.append(PathItem(child, disk_path))
                logger.info(f"Detected new output item in mem: {child.name}")

            # Remove pending items that have been deleted.
            for item in pending_items[:]:
                if not item.path.exists() and item not in inflight.values():
                    item.delete_from_disk()
                    pending_items.remove(item)
                    logger.warning(f"Pending item deleted before or during sync: {item.name}")

            # Exit condition: no pending, no inflight, and training done.
            if len(pending_items) == 0 and len(inflight) == 0 and training_finished:
                logger.info("Training process finished and no pending items left; exiting writer loop.")
                break

            # Backpressure: pause training process if needed.
            n_pending_dirs = sum(1 for p in pending_items if p.path.is_dir()) + sum(
                1 for p in inflight.values() if p.path.is_dir()
            )
            shm_used = _statvfs_used_percent(mem_path)
            if not training_proc.paused and should_pause(n_pending_dirs, shm_used):
                logger.warning(
                    f"Backpressure: pausing training (SIGSTOP), pending_dirs={n_pending_dirs}, shm_used={shm_used:.1f}%"
                )
                training_proc.pause()

            # Launch sync tasks in background threads (non-blocking for main loop)
            slots = max(sync_concurrency - len(inflight), 0)
            if slots > 0:
                now = time.time()
                inflight_names = {p.name for p in inflight.values()}
                stable_items = [
                    item
                    for item in pending_items
                    if item.name not in inflight_names
                    and item.is_stable(now, checkpoint_stable_sec, training_proc.process)
                ]
                stable_items.sort(key=lambda x: (not x.is_dir, x.last_change_ts), reverse=True)  # Newest directory first
                for item in stable_items[:slots]:
                    task = asyncio.create_task(asyncio.to_thread(item.sync_to_disk))
                    inflight[task] = item

            # Backpressure: resume training process if needed.
            n_pending_dirs = sum(1 for p in pending_items if p.path.is_dir()) + sum(
                1 for p in inflight.values() if p.path.is_dir()
            )
            shm_used = _statvfs_used_percent(mem_path)
            if training_proc.paused and training_proc.is_running():
                if should_resume(n_pending_dirs, shm_used):
                    logger.info(
                        f"Backpressure: resuming training (SIGCONT), pending_dirs={n_pending_dirs}, shm_used={shm_used:.1f}%"
                    )
                    training_proc.resume()

            status = (
                f"Writer status: synced={len(synced_items)}, pending={len(pending_items)} syncing={len(inflight)}/{sync_concurrency}"
                f" shm_used={shm_used:.1f}% pid={training_proc.pid} running={training_proc.is_running()} paused={training_proc.paused}"
            )
            if ui.enabled:
                ui.set_status(status)
            else:
                logger.info(status)

            await asyncio.sleep(poll_interval_sec)

        # Delete synced items that have been removed from mem_dir.
        for item in synced_items[:]:
            if not item.path.exists():
                item.delete_from_disk()
                synced_items.remove(item)
                logger.info(f"Synced item deleted from mem: {item.name}, so delete it from disk too.")

        logger.info("Async output writer exited")


def _remove_any(p: Path) -> None:
    try:
        if p.is_symlink() or p.is_file():
            p.unlink(missing_ok=True)
        elif p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink(missing_ok=True)
    except Exception:
        pass


def _statvfs_used_percent(path: Path) -> float:
    st = os.statvfs(path)
    total = st.f_blocks * st.f_frsize
    free = st.f_bfree * st.f_frsize
    used = max(total - free, 0)
    if total <= 0:
        return 0.0
    return used * 100.0 / total


if __name__ == "__main__":
    import fire

    fire.Fire(run)
