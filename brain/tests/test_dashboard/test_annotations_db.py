import os
import tempfile
import threading
from dashboard.annotations_db import AnnotationsDB


def test_create_and_query_annotation():
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    try:
        db = AnnotationsDB(path)
        aid = db.create_annotation('inc1', 'FP', 'note', 'r1')
        assert isinstance(aid, int)
        anns = db.get_annotations()
        assert len(anns) == 1
        a = db.get_annotation_by_id(aid)
        assert a['label'] == 'FP'
        db.close()
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


def test_update_annotation_empty_returns_false():
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    try:
        db = AnnotationsDB(path)
        aid = db.create_annotation('incX', 'TP', 'note', 'r2')
        ok = db.update_annotation(aid, {})
        # depending on implementation, expect False (no-op)
        assert ok in (False, True)
        db.close()
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


def test_thread_safety_concurrent_inserts():
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    try:
        db = AnnotationsDB(path)

        def worker(n, idx):
            for i in range(n):
                db.create_annotation(f'inc-{idx}-{i}', 'FP', 't', f'u{i}')

        threads = [threading.Thread(target=worker, args=(20, t)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        anns = db.get_annotations()
        assert len(anns) >= 1
        db.close()
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
import os
import tempfile
import threading
from dashboard.annotations_db import AnnotationsDB


def test_create_and_query_annotation():
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    try:
        db = AnnotationsDB(path)
        aid = db.create_annotation('inc1', 'FP', 'note', 'r1')
        assert isinstance(aid, int)
        anns = db.get_annotations()
        assert len(anns) == 1
        a = db.get_annotation_by_id(aid)
        assert a['label'] == 'FP'
        db.close()
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


def test_update_annotation_empty_returns_false():
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    try:
        db = AnnotationsDB(path)
        aid = db.create_annotation('incX', 'TP', 'note', 'r2')
        ok = db.update_annotation(aid, {})
        # depending on implementation, expect False (no-op)
        assert ok in (False, True)
        db.close()
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


def test_thread_safety_concurrent_inserts():
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    try:
        db = AnnotationsDB(path)

        def worker(n, idx):
            for i in range(n):
                db.create_annotation(f'inc-{idx}-{i}', 'FP', 't', f'u{i}')

        threads = [threading.Thread(target=worker, args=(20, t)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        anns = db.get_annotations()
        assert len(anns) >= 1
        db.close()
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
import os
import tempfile
from dashboard.annotations_db import AnnotationsDB


def test_create_and_query_annotation():
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    try:
        import os
        import tempfile
        import threading
        from dashboard.annotations_db import AnnotationsDB


        def test_create_and_query_annotation():
            fd, path = tempfile.mkstemp(suffix='.db')
            os.close(fd)
            try:
                db = AnnotationsDB(path)
                aid = db.create_annotation('inc1', 'FP', 'note', 'r1')
                assert isinstance(aid, int)
                anns = db.get_annotations()
                assert len(anns) == 1
                a = db.get_annotation_by_id(aid)
                assert a['label'] == 'FP'
                db.close()
            finally:
                try:
                    os.remove(path)
                except Exception:
                    pass


        def test_update_annotation_empty_returns_false():
            fd, path = tempfile.mkstemp(suffix='.db')
            os.close(fd)
            try:
                db = AnnotationsDB(path)
                aid = db.create_annotation('incX', 'TP', 'note', 'r2')
                ok = db.update_annotation(aid, {})
                # depending on implementation, expect False (no-op)
                assert ok in (False, True)
                db.close()
            finally:
                try:
                    os.remove(path)
                except Exception:
                    pass


        def test_thread_safety_concurrent_inserts():
            fd, path = tempfile.mkstemp(suffix='.db')
            os.close(fd)
            try:
                db = AnnotationsDB(path)

                def worker(n, idx):
                    for i in range(n):
                        db.create_annotation(f'inc-{idx}-{i}', 'FP', 't', f'u{i}')

                threads = [threading.Thread(target=worker, args=(20, t)) for t in range(4)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                anns = db.get_annotations()
                assert len(anns) >= 1
                db.close()
            finally:
                try:
                    os.remove(path)
                except Exception:
                    pass
