"""Headless quickstart: load a precomputed graph and run iSTAMP inference."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "reanalysis"))
import _lib_raman as L  # noqa
def main():
    model, *_ = L.load_model()
    import glob
    f = sorted(glob.glob(os.path.join(L.DATA, "Img_*_pyg.pt")))[:1]
    if not f:
        print("No example graph found; set L.DATA to your graph directory."); return
    d = L.load_graph_dict(f[0]); g = d["pyg_dataset_by_hop"][3][0]
    pred, n = L.infer_graph(model, g)
    sm = L.softmax(pred)
    print("image:", d["image_id"], "nodes:", n)
    for c, p in zip(L.CLASSES, sm):
        print(f"  {c:14s} {p:.3f}")
    print("predicted:", L.CLASSES[int(sm.argmax())])
if __name__ == "__main__":
    main()
