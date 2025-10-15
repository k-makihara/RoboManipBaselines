# suggest_geom_pos.py
import argparse, trimesh

def mesh_bottom(path):
    m = trimesh.load(path, force='mesh')
    if not isinstance(m, trimesh.Trimesh):
        m = trimesh.util.concatenate([g for g in m.geometry.values()])
    return float(m.bounds[0][2]), m.bounds, m.centroid  # min_z, AABB, centroid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visual", required=True, help="visual mesh path (.obj)")
    ap.add_argument("--collision", required=True, help="collision mesh path (.stl/.obj)")
    ap.add_argument("--pv", type=float, nargs=3, default=[0,0,0], help="visual geom.pos (x y z)")
    ap.add_argument("--pc", type=float, nargs=3, default=[0,0,0], help="collision geom.pos (x y z)")
    ap.add_argument("--sv", type=float, nargs=3, default=[1,1,1], help="visual mesh scale (sx sy sz)")
    ap.add_argument("--sc", type=float, nargs=3, default=[1,1,1], help="collision mesh scale (sx sy sz)")
    ap.add_argument("--mode", choices=["bottom","centroid"], default="bottom")
    args = ap.parse_args()

    bV, aabbV, cV = mesh_bottom(args.visual)
    bC, aabbC, cC = mesh_bottom(args.collision)

    pvx, pvy, pvz = args.pv
    pcx, pcy, pcz = args.pc
    svx, svy, svz = args.sv
    scx, scy, scz = args.sc

    if args.mode == "bottom":
        dz = (bV*svz + pvz) - (bC*scz + pcz)
        dx = 0.0
        dy = 0.0
        why = "match bottom (table contact)"
    else:  # centroid合わせ（水平位置を揃えたい時）
        dz = ((cV[2]*svz + pvz) - (cC[2]*scz + pcz))
        dx = ((cV[0]*svx + pvx) - (cC[0]*scx + pcx))
        dy = ((cV[1]*svy + pvy) - (cC[1]*scy + pcy))
        why = "match centroids (xy & z)"

    print("=== SUGGESTED geom.pos FOR COLLISION ===")
    print(f"current p_c = ({pcx:.6f}, {pcy:.6f}, {pcz:.6f})")
    print(f"add Δ = ({dx:.6f}, {dy:.6f}, {dz:.6f})  # {why}")
    print(f"--> new p_c = ({pcx+dx:.6f}, {pcy+dy:.6f}, {pcz+dz:.6f})")

if __name__ == "__main__":
    main()
