#![allow(dead_code)]

#[allow(unused_imports)]
use std::f32::consts::PI;

use raylib::prelude::*;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

const SIZE: i32 = 256;
const N: usize = SIZE as usize;
const SCALE: i32 = 4;

fn draw_square(d: &mut RaylibDrawHandle, x: i32, y: i32, color: Color) {
    d.draw_rectangle(x * SCALE, y * SCALE, SCALE, SCALE, color);
}

type Array2D = Box<[[f32; N]; N]>;

struct Fluid {
    dt: f32,
    diff: f32,
    visc: f32,

    s: Array2D,
    density_r: Array2D,
    density_g: Array2D,
    density_b: Array2D,

    vx: Array2D,
    vy: Array2D,

    vx0: Array2D,
    vy0: Array2D,

    fade_amt: f32,
}

impl Fluid {
    fn new(dt: f32, diff: f32, visc: f32) -> Fluid {
        Fluid {
            dt,
            diff,
            visc,

            s: Box::new([[0.0; N]; N]),
            density_r: Box::new([[0.0; N]; N]),
            density_g: Box::new([[0.0; N]; N]),
            density_b: Box::new([[0.0; N]; N]),

            vx: Box::new([[0.0; N]; N]),
            vy: Box::new([[0.0; N]; N]),
            vx0: Box::new([[0.0; N]; N]),
            vy0: Box::new([[0.0; N]; N]),

            fade_amt: 0.99,
        }
    }

    fn render_den(&self, d: &mut RaylibDrawHandle) {
        for i in 0..N {
            for j in 0..N {
                let color_r = (self.density_r[i][j] * 255.0) as u8;
                let color_g = (self.density_g[i][j] * 255.0) as u8;
                let color_b = (self.density_b[i][j] * 255.0) as u8;
                let color = Color::new(color_r, color_g, color_b, 255);
                draw_square(d, i as i32, j as i32, color);
            }
        }
    }

    fn render_vel(&self, d: &mut RaylibDrawHandle) {
        for i in 0..N {
            for j in 0..N {
                let color_val_x = (self.vx[i][j] * 255.0) as u8;
                let color_val_y = (self.vy[i][j] * 255.0) as u8;

                let color = Color::new(color_val_x, color_val_y, 0, 255);
                draw_square(d, i as i32, j as i32, color);
            }
        }
    }

    fn add_density(&mut self, mut x: usize, mut y: usize, amt_r: f32, amt_g: f32, amt_b: f32) {
        x = x.min(N - 1);
        y = y.min(N - 1);
        self.density_r[x][y] += amt_r;
        self.density_g[x][y] += amt_g;
        self.density_b[x][y] += amt_b;
    }

    fn add_velocity(&mut self, mut x: usize, mut y: usize, amt_x: f32, amt_y: f32) {
        x = x.min(N - 1);
        y = y.min(N - 1);

        self.vx[x][y] += amt_x;
        self.vy[x][y] += amt_y;
    }

    fn step(&mut self) {
        const ITER: i32 = 16;

        diffuse(1, &mut self.vx0, &mut self.vx, self.visc, self.dt, ITER);
        diffuse(2, &mut self.vy0, &mut self.vy, self.visc, self.dt, ITER);

        project(
            &mut self.vx0,
            &mut self.vy0,
            &mut self.vx,
            &mut self.vy,
            ITER,
        );

        advect(
            1,
            &mut self.vx,
            Axis::X,
            &mut self.vx0,
            &mut self.vy0,
            self.dt,
        );
        advect(
            2,
            &mut self.vy,
            Axis::Y,
            &mut self.vx0,
            &mut self.vy0,
            self.dt,
        );

        project(
            &mut self.vx,
            &mut self.vy,
            &mut self.vx0,
            &mut self.vy0,
            ITER,
        );

        diffuse(
            0,
            &mut self.s,
            &mut self.density_r,
            self.diff,
            self.dt,
            ITER,
        );
        advect2(
            0,
            &mut self.density_r,
            &mut self.s,
            &mut self.vx,
            &mut self.vy,
            self.dt,
        );

        diffuse(
            0,
            &mut self.s,
            &mut self.density_g,
            self.diff,
            self.dt,
            ITER,
        );
        advect2(
            0,
            &mut self.density_g,
            &mut self.s,
            &mut self.vx,
            &mut self.vy,
            self.dt,
        );

        diffuse(
            0,
            &mut self.s,
            &mut self.density_b,
            self.diff,
            self.dt,
            ITER,
        );
        advect2(
            0,
            &mut self.density_b,
            &mut self.s,
            &mut self.vx,
            &mut self.vy,
            self.dt,
        );

        fade(&mut self.density_r, self.fade_amt);
        fade(&mut self.density_g, self.fade_amt);
        fade(&mut self.density_b, self.fade_amt);

        set_bnd(1, &mut self.vx);
        set_bnd(2, &mut self.vy);
        set_bnd(0, &mut self.density_r);
        set_bnd(0, &mut self.density_g);
        set_bnd(0, &mut self.density_b);
    }
}

#[rustfmt::skip]
fn set_bnd(b: usize, x: &mut Array2D) {
    for i in 1..(N - 1) {
        x[i][0] = if b == 2 { -x[i][1] } else { x[i][1] };
        x[i][N - 1] = if b == 2 { -x[i][N - 2] } else { x[i][N - 2] };
    }

    for j in 1..(N - 1) {
        x[0][j] =   if b == 1 { -x[1][j]     } else { x[1][j]     };
        x[N-1][j] = if b == 1 { -x[N - 2][j] } else { x[N - 2][j] };
    }

    x[0][0]     = 0.5 * (x[1][0]     + x[0][1]    );
    x[0][N-1]   = 0.5 * (x[1][N-1]   + x[0][N-2]  );
    x[N-1][0]   = 0.5 * (x[N-1][1]   + x[N-2][0]  );
    x[N-1][N-1] = 0.5 * (x[N-2][N-1] + x[N-1][N-2]);
}

#[rustfmt::skip]
fn lin_solve(b: usize, x: &mut Array2D, x0: &mut Array2D, a: f32, c: f32, iter: i32) {
    let c_recip = 1.0 / c;

    for _ in 0..iter {
        for i in 1..(N - 1) {
            for j in 1..(N - 1) {
                x[i][j] = (x0[i][j] + 
                    a * 
                    (
                    x[i + 1][j] + 
                    x[i - 1][j] + 
                    x[i][j + 1] + 
                    x[i][j - 1])
                    ) * c_recip;
            }
        }

        set_bnd(b, x);
    }
}

fn diffuse(b: usize, x: &mut Array2D, x0: &mut Array2D, diff: f32, dt: f32, iter: i32) {
    let a = dt * diff * ((N - 2) * (N - 2)) as f32;
    lin_solve(b, x, x0, a, 1.0 + 4.0 * a, iter);
}

#[rustfmt::skip]
fn project(vx: &mut Array2D, vy: &mut Array2D, p: &mut Array2D, div: &mut Array2D, iter: i32) {
    p[1..(N-1)].iter_mut().for_each(|row| row[1..(N-1)].fill(0.0));

    div[1..(N-1)].par_iter_mut().enumerate().for_each(|(i,line)| line[1..(N-1)].par_iter_mut().enumerate().for_each(|(j,div)| {
        *div = -0.5 * (
                vx[i+2][j+1] - vx[i][j+1] +
                vy[i+1][j+2] - vy[i+1][j]
            )/N as f32;
    } ));

    set_bnd(0, div);
    set_bnd(0, p);
    lin_solve(0, p, div, 1.0, 4.0, iter);

    for i in 1..(N - 1) {
        for j in 1..(N - 1) {
            vx[i][j] -= 0.5 * (p[i + 1][j] - p[i - 1][j]) * N as f32;
            vy[i][j] -= 0.5 * (p[i][j + 1] - p[i][j - 1]) * N as f32;
            
        }
    }
    set_bnd(1, vx);
    set_bnd(2, vy);
}

fn fade(d: &mut Array2D, fade_amt: f32) {
    for cell in d.iter_mut().flatten() {
        *cell *= fade_amt;
    }
}

fn clamp(num: f32, low: f32, high: f32) -> f32 {
    num.max(low).min(high)
}

#[derive(Debug, Clone, Copy)]
enum Axis {
    X,
    Y,
}

#[rustfmt::skip]
fn advect<'a>(
    b: usize,
    d: &mut Array2D,
    axis: Axis,
    vx: &'a mut Array2D,
    vy: &'a mut Array2D,
    dt: f32,
) {
    let dtx = dt * (N - 2) as f32;
    let dty = dt * (N - 2) as f32;

    let n_float = N as f32;

    let (mut i0, mut i1, mut j0, mut j1);
    let (mut tmp1, mut tmp2, mut x, mut y);
    let (mut s0, mut s1, mut t0, mut t1);

    for i in 1..(N - 1) {
        for j in 1..(N - 1) {
            tmp1 = dtx * vx[i][j];
            tmp2 = dty * vy[i][j];

            x = i as f32 - tmp1;
            y = j as f32 - tmp2;

            x = clamp(x, 0.5, n_float + 0.5);
            i0 = x.floor();
            i1 = i0 + 1.0;

            y = clamp(y, 0.5, n_float + 0.5);
            j0 = y.floor();
            j1 = j0 + 1.0;

            s1 = x - i0;
            s0 = 1.0 - s1;
            t1 = y - j0;
            t0 = 1.0 - t1;

            let i0i = (i0 as usize).min(N-1);
            let i1i = (i1 as usize).min(N-1);
            let j0i = (j0 as usize).min(N-1);
            let j1i = (j1 as usize).min(N-1);

            match axis {
                Axis::X => {
                    d[i][j] = s0 * (
                        t0 * vx[i0i][j0i]+ 
                        t1 * vx[i0i][j1i]
                        ) + s1 * (
                        t0 * vx[i1i][j0i] + 
                        t1 * vx[i1i][j1i]
                    );
                },
                Axis::Y => {
                    d[i][j] = s0 * (t0 * vy[i0i][j0i] + t1 * vy[i0i][j1i]) +
                              s1 * (t0 * vy[i1i][j0i] + t1 * vy[i1i][j1i]);
                }
            };

        }
    }

    set_bnd(b, d);
}

fn advect2<'a>(
    b: usize,
    d: &mut Array2D,
    d0: &mut Array2D,
    vx: &'a mut Array2D,
    vy: &'a mut Array2D,
    dt: f32,
) {
    let dtx = dt * (N - 2) as f32;
    let dty = dt * (N - 2) as f32;

    let n_float = N as f32;

    let (mut i0, mut i1, mut j0, mut j1);
    let (mut tmp1, mut tmp2, mut x, mut y);
    let (mut s0, mut s1, mut t0, mut t1);

    for i in 1..(N - 1) {
        for j in 1..(N - 1) {
            tmp1 = dtx * vx[i][j];
            tmp2 = dty * vy[i][j];

            x = i as f32 - tmp1;
            y = j as f32 - tmp2;

            x = clamp(x, 0.5, n_float + 0.5);
            i0 = x.floor();
            i1 = i0 + 1.0;

            y = clamp(y, 0.5, n_float + 0.5);
            j0 = y.floor();
            j1 = j0 + 1.0;

            s1 = x - i0;
            s0 = 1.0 - s1;
            t1 = y - j0;
            t0 = 1.0 - t1;

            let i0i = (i0 as usize).min(N - 1);
            let i1i = (i1 as usize).min(N - 1);
            let j0i = (j0 as usize).min(N - 1);
            let j1i = (j1 as usize).min(N - 1);

            d[i][j] = s0 * (t0 * d0[i0i][j0i] + t1 * d0[i0i][j1i])
                + s1 * (t0 * d0[i1i][j0i] + t1 * d0[i1i][j1i]);
        }
    }

    set_bnd(b, d);
}

const FPS: u32 = 60;

#[allow(unused)]
fn main() {
    let (mut rl, thread) = raylib::init()
        .size(SIZE * SCALE, SIZE * SCALE)
        .title("Fluid")
        .build();

    rl.set_target_fps(FPS);

    let mut fluid = Fluid::new(1.0 / FPS as f32, 0.001, 0.000001);

    let mut angle = 0.0;

    let mut mouse_pos;
    let mut mouse_delta;
    let mut prev_mouse_pos;

    let mut color_idx = 0;

    let colors = [
        (1.0, 2.0, 2.0),
        (1.0, 2.0, 1.0),
        (2.0, 2.0, 1.0),
        (2.0, 1.0, 2.0),
        (2.0, 1.5, 1.0),
    ];

    let color_factor = 2.0;
    while !rl.window_should_close() {
        mouse_pos = rl.get_mouse_position();
        mouse_delta = rl.get_mouse_delta();
        prev_mouse_pos = mouse_pos - (mouse_delta);

        let color = colors[color_idx];

        angle += 0.1 * rand::random_range(-1.0..1.0f32);
        let vel_x = angle.cos();
        let vel_y = angle.sin();

        for pair in [
            (N / 2, N / 2),
            (N / 2 - 1, N / 2),
            (N / 2 + 1, N / 2),
            (N / 2, N / 2 - 1),
            (N / 2, N / 2 + 1),
            (N / 2 - 1, N / 2 - 1),
            (N / 2 - 1, N / 2 + 1),
            (N / 2 + 1, N / 2 - 1),
            (N / 2 + 1, N / 2 + 1),
        ] {
            fluid.add_density(
                pair.0,
                pair.1,
                color.0 * color_factor * 0.4,
                color.1 * color_factor * 0.4,
                color.2 * color_factor * 0.4,
            );
            fluid.add_velocity(pair.0, pair.1, vel_x * 0.5, vel_y * 0.5);
        }

        if rl.is_key_pressed(KeyboardKey::KEY_SPACE) {
            color_idx = (color_idx + 1) % colors.len();
        }

        if rl.is_key_pressed(KeyboardKey::KEY_A) {
            if fluid.fade_amt == 0.99 {
                fluid.fade_amt = 0.999;
                println!("Low fade.");
            } else {
                fluid.fade_amt = 0.99;
                println!("High hade.");
            }
        }

        if rl.is_mouse_button_down(MouseButton::MOUSE_BUTTON_LEFT) {
            for pair in [
                (
                    mouse_pos.x as usize / SCALE as usize,
                    mouse_pos.y as usize / SCALE as usize,
                ),
                (
                    mouse_pos.x as usize / SCALE as usize + 1,
                    mouse_pos.y as usize / SCALE as usize,
                ),
                (
                    mouse_pos.x as usize / SCALE as usize,
                    mouse_pos.y as usize / SCALE as usize + 1,
                ),
                (
                    mouse_pos.x as usize / SCALE as usize + 1,
                    mouse_pos.y as usize / SCALE as usize + 1,
                ),
            ] {
                fluid.add_density(
                    pair.0,
                    pair.1,
                    color.0 * color_factor,
                    color.1 * color_factor,
                    color.2 * color_factor,
                );
            }

            fluid.add_velocity(
                mouse_pos.x as usize / SCALE as usize,
                mouse_pos.y as usize / SCALE as usize,
                mouse_delta.x,
                mouse_delta.y,
            );
        }

        if rl.is_mouse_button_down(MouseButton::MOUSE_BUTTON_RIGHT) {
            fluid.add_velocity(
                mouse_pos.x as usize / SCALE as usize,
                mouse_pos.y as usize / SCALE as usize,
                mouse_delta.x,
                mouse_delta.y,
            );
        }

        fluid.step();

        let mut d = rl.begin_drawing(&thread);

        d.clear_background(Color::BLACK);
        fluid.render_den(&mut d);
    }
}
