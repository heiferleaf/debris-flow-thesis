"""One explicitly assumed river geometry for testing physical video coupling."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from physics_core import G, advance, timestep, verify


def run(inp, output, config=None):
    cfg=json.loads(config.read_text('utf8')) if config else {}
    output.mkdir(parents=True, exist_ok=True)
    if (output/'simulation.npz').exists():raise ValueError('Preserve previous simulation')
    checks = verify()
    print('core_checks ' + json.dumps(checks), flush=True)
    image = cv2.imread(str(inp/'wet-first-frame.png'))
    source_mask = cv2.imread(str(inp/'water-mask.png'), 0)
    # Pixel correspondences selected from visible inlet/outlet edges. Metric
    # dimensions are scenario assumptions, NOT recovered scene measurements.
    image_quad = np.array(cfg.get('image_quad',[[50,719],[780,719],[1279,486],[1279,444]]), np.float32)
    channel_width=cfg.get('channel_width_m',14.);channel_length=cfg.get('channel_length_m',60.)
    ground_quad = np.array([[0,0],[channel_width,0],[channel_width,channel_length],[0,channel_length]], np.float32)
    img_to_world = cv2.getPerspectiveTransform(image_quad, ground_quad)
    dx=cfg.get('grid_cell_m',.4);margin=cfg.get('lateral_margin_m',4.)
    nx=round((channel_width+2*margin)/dx);ny=round(channel_length/dx)
    world_to_grid = np.array([[1/dx,0,margin/dx-.5],[0,1/dx,-.5],[0,0,1]], float)
    img_to_grid = world_to_grid @ img_to_world
    mask = cv2.warpPerspective(source_mask, img_to_grid, (nx,ny), flags=cv2.INTER_NEAREST) > 0
    count, components = cv2.connectedComponents(mask.astype(np.uint8))
    mask = components == (1 + np.argmax(np.bincount(components.ravel())[1:]))
    if not mask[0].any() or not mask[-1].any():
        raise ValueError('Projected geometry has no open inlet/outlet')
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5) * dx
    yy, xx = np.mgrid[:ny,:nx]
    slope = cfg.get('slope',.002);bank_rise=cfg.get('bank_bed_rise_m',.55)
    bank_scale=cfg.get('bank_length_scale_m',1.2);central_depth=cfg.get('central_depth_m',.8)
    velocity=cfg.get('inlet_velocity_m_per_s',.65);viscosity=cfg.get('eddy_viscosity_m2_per_s',.03);manning=cfg.get('manning_n',.035)
    bed = -slope*(yy+.5)*dx + bank_rise*np.exp(-distance/bank_scale)
    stage = central_depth - slope*(yy+.5)*dx
    h = np.maximum(stage-bed, .06)
    state = np.zeros((ny,nx,3),float)
    state[...,0] = h
    state[...,2] = h*velocity
    state[~mask] = 0
    boundary = dict(kind='channel_y', inflow_per_width=h[0]*velocity,
                    outlet_stage=central_depth-slope*(ny-.5)*dx)
    cv2.imwrite(str(output/'ground-water-mask.png'),mask.astype(np.uint8)*255)
    cv2.imwrite(str(output/'reference.png'),image)
    initial_volume = float(state[...,0].sum()*dx*dx)
    accumulated_flux = 0.; t = 0.; steps = 0; warmup = cfg.get('warmup_seconds',45.); duration = 12.; fps = 24
    times = warmup + np.arange(round(duration*fps))/fps
    states = []; started = time.time(); last_report = -1
    for target in times:
        while t < target-1e-10:
            dt = min(timestep(state,mask,dx,viscosity), target-t)
            state, net = advance(state,bed,mask,dx,dt,boundary,t,manning,viscosity)
            accumulated_flux += net; t += dt; steps += 1
            report=int(t//10)
            if report!=last_report:
                last_report=report
                print(f'simulated {t:.2f}s elapsed {time.time()-started:.1f}s',flush=True)
        states.append(state.astype(np.float32))
    volume = float(state[...,0].sum()*dx*dx)
    residual = volume-initial_volume-accumulated_flux
    relative = abs(residual)/initial_volume
    if relative > 1e-8:raise ValueError('Open-boundary water budget mismatch')
    stack = np.stack(states)
    np.savez_compressed(output/'simulation.npz',states=stack,bed=bed.astype(np.float32),mask=mask,
                        img_to_world=img_to_world,img_to_grid=img_to_grid,dx=dx,times=times)
    v = np.linalg.norm(state[...,1:],axis=-1)/np.maximum(state[...,0],1e-8)
    stats = dict(min_depth_m=float(state[...,0][mask].min()),max_depth_m=float(state[...,0][mask].max()),
        max_depth_average_speed_m_per_s=float(v[mask].max()),
        near_bank_mean_speed_m_per_s=float(v[mask&(distance<=1.2)].mean()),
        interior_mean_speed_m_per_s=float(v[mask&(distance>1.2)].mean()),
        initial_volume_m3=initial_volume,final_volume_m3=volume,integrated_net_boundary_volume_m3=accumulated_flux,
        mass_budget_relative_residual=relative)
    record = dict(status='scenario_simulation_finished_not_site_calibrated',source=str(inp),
        source_frame_sha256=hashlib.sha256((inp/'wet-first-frame.png').read_bytes()).hexdigest(),
        assumptions=dict(image_quad=image_quad.tolist(),ground_quad_m=ground_quad.tolist(),
            central_depth_m=central_depth,bank_bed_rise_m=bank_rise,bank_length_scale_m=bank_scale,slope=slope,
            manning_n=manning,eddy_viscosity_m2_per_s=viscosity,inlet_nominal_velocity_m_per_s=velocity,
            inflow_variation='2.5% / 7s and 1.5% / 2.3s sine perturbations',
            warmup_seconds=warmup,grid_cell_m=dx,measured_site_parameters=False,
            configuration=str(config) if config else None,configuration_values=cfg),
        core_checks=checks,stats=stats,frames=len(states),fps=fps,elapsed_seconds=time.time()-started,
        solver_steps=steps,physical_model='2D depth-averaged shallow water with Manning and lateral momentum diffusion',
        limitations=['assumed ground-plane homography and bed geometry','first-order spatial scheme',
            'does not resolve 3D splashes, foam or capillary waves','surface rendering still to be validated'],
        user_accepted=False,training_ready=False)
    (output/'simulation.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    print(json.dumps(stats),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path)
    a=p.parse_args();run(a.input,a.output,a.config)
