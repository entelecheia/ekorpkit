import subprocess

def capture(command):
    proc = subprocess.Popen(command,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
    )
    out,err = proc.communicate()
    return out, err, proc.returncode


def test_ekorpkit_no_param():
    command = ["ekorpkit"]
    out, err, exitcode = capture(command)
    assert exitcode == 0

def test_ekorpkit_default():
    command = ["ekorpkit", "cmd=default"]
    out, err, exitcode = capture(command)
    assert exitcode == 0

def test_ekorpkit_listup():
    command = ["ekorpkit", "cmd=listup"]
    out, err, exitcode = capture(command)
    assert exitcode == 0
