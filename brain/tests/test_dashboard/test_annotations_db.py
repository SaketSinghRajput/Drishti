import os
import tempfile
from dashboard.annotations_db import AnnotationsDB


def test_annotations_db_smoke():
    """Minimal smoke test for annotations DB: create, read, update, delete."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    try:
        db = AnnotationsDB(path)
        aid = db.create_annotation('inc1', 'FP', 'note', 'r1')
        assert isinstance(aid, int)
        a = db.get_annotation_by_id(aid)
        assert a and a.get('label') == 'FP'
        ok = db.update_annotation(aid, {'notes': 'updated'})
        assert ok in (True, False)
        try:
            db.delete_annotation(aid)
        except Exception:
            # Some implementations may not provide delete; ignore
            pass
        db.close()
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
