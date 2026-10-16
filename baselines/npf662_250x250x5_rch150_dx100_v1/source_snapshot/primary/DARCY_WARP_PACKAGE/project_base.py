from shutil import which
import warnings
import platform
os_type = platform.system()
import socket
from pathlib import Path

print('running on host:', socket.gethostname())
project_name = 'canterbury_nn'
proj_root = Path(__file__).parent  # base of git repo
home = Path.home()

data_store = proj_root.joinpath("data")
# make sure the wq_data store exists else create it
if not data_store.exists():
    data_store.mkdir(exist_ok=True)

# get opperating system type eg windows or linux
if os_type == "Windows":
    unc = Path(r"\\DS1621\Environment_Moulding_Storage\PINN")
    print(unc.exists())  # should be True
else:
    unc = home.joinpath('mnt','unbacked','PINN')
    print(unc.exists())

_MF6_ERROR = (
    "mf6 not found. Either add /bin/modflow to PATH for your Python environment/IDE, "
    "or install the binary to a standard location. "
    "Tried '/bin/modflow/mf6' and PATH."
)

MF6: Path | None = Path("/bin/modflow/mf6")
if not MF6.exists():
    found = which("mf6")
    if found is None:
        MF6 = None
    else:
        MF6 = Path(found)

def require_mf6() -> Path:
    if MF6 is None:
        raise RuntimeError(_MF6_ERROR)
    return MF6
