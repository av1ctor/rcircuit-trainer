use bot_engine_gltf::renderer::{Renderer, ModelRef, RGB};
use sdl2::{pixels, render::{CanvasBuilder, Canvas}, video::Window, EventPump, rect::Point};
use three_rs::math::{vector3::Vector3, triangle::Triangle, quaternion::Quaternion};

use crate::loaders::glb;

pub struct Object {
    pub visible: bool,
    pub pos: Vector3,
    pub rot: Vector3,
    pub q: Quaternion,
    pub scale: Vector3,
    pub triangles: Vec<Triangle>,
    pub is_mesh: bool,
    pub is_open: bool,
    pub color: RGB,
}

impl Object {
    pub fn new(
        triangles: Vec<Triangle>,
        is_mesh: bool,
        is_open: bool,
        color: RGB
    ) -> Self {
        Self {
            pos: Vector3::zero(),
            rot: Vector3::zero(),
            q: Quaternion::identity(),
            scale: Vector3::one(),
            visible: false,
            triangles,
            is_mesh,
            is_open,
            color,
        }
    }
}

pub struct SdlRenderer {
    pub w: usize,
    pub h: usize,
    pub canvas: Canvas<Window>,
    pub objects: Vec<Object>,
}

impl SdlRenderer {
    pub fn new(
        w: u32,
        h: u32
    ) -> Result<(Self, EventPump), String> {
        let sdl = sdl2::init().unwrap();
        let video = sdl.video().unwrap();
        let window = video
            .window("trainer", w, h)
            .build()
            .unwrap();
        let event_loop = sdl.event_pump().unwrap();
        let canvas = CanvasBuilder::new(window)
            .build()
            .unwrap();

        Ok((Self{
            w: w as _,
            h: h as _,
            canvas, 
            objects: vec![],
        }, event_loop))
    }
}

impl Renderer for SdlRenderer {
    fn reset(
        &mut self
    ) {
        self.objects = vec![];
    }
    
    fn add_model_from_file(
        &mut self,
        color: RGB,
        id: &str
    ) -> Result<ModelRef, String> {
        let triangles = glb::load_from_file(id)?;
        let rf = self.objects.len();

        self.objects.push(Object::new(triangles, true, false, color));

        Ok(rf)
    }

    fn add_model_from_bytes(
        &mut self,
        color: RGB,
        bytes: &[u8]
    ) -> Result<ModelRef, String> {
        let triangles = glb::load_from_bytes(bytes)?;
        let rf = self.objects.len();

        self.objects.push(Object::new(triangles, true, false, color));

        Ok(rf)
    }

    fn add_model_from_gltf(
        &mut self,
        color: RGB,
        doc: &gltf::Document,
        buffers: &Vec<gltf::buffer::Data>
    ) -> Result<ModelRef, String> {
        let triangles = glb::load(&doc.default_scene().unwrap(), buffers)?;
        let rf = self.objects.len();

        self.objects.push(Object::new(triangles, true, false, color));

        Ok(rf)
    }

    fn add_model_from_triangles(
        &mut self,
        color: RGB,
        triangles: Vec<Triangle>,
        is_open: bool
    ) -> Result<ModelRef, String> {
        let rf = self.objects.len();

        self.objects.push(Object::new(triangles, false, is_open, color));

        Ok(rf)
    }

    fn show_model(
        &mut self,
        rf: ModelRef
    ) {
        self.objects[rf].visible = true;
    }

    fn hide_model(
        &mut self,
        rf: ModelRef
    ) {
        self.objects[rf].visible = false;
    }

    fn set_position(
        &mut self,
        rf: ModelRef,
        pos: &Vector3
    ) {
        self.objects[rf].pos = pos.clone();
    }

    fn set_rotation(
        &mut self,
        rf: ModelRef,
        rot: &Vector3,
        q: &Quaternion
    ) {
        self.objects[rf].rot = rot.clone();
        self.objects[rf].q = q.clone();
    }

    fn set_scale(
        &mut self,
        rf: ModelRef,
        scale: &Vector3
    ) {
        self.objects[rf].scale = scale.clone();
    }

    fn render(
        &mut self
    ) {
        self.canvas.set_draw_color(pixels::Color::RGB(0, 0, 0));
        self.canvas.clear();

        let a = (self.w as f32) / (self.h as f32);
        let w = (self.w as f32) / 80.0;
        let h = (self.h as f32) * a / 80.0;

        let hw = 200.0;
        let hh = 200.0;

        for obj in &self.objects {
            if obj.visible {
                self.canvas.set_draw_color(
                    pixels::Color::RGB(obj.color[0], obj.color[1], obj.color[2])
                );

                for tri in &obj.triangles {
                    _ = self.canvas.draw_line(
                        Point::new(
                            (hw + (obj.pos.x + tri.a.x) * w) as i32, 
                            (hh + (obj.pos.z + tri.a.z) * h) as i32
                        ), 
                        Point::new(
                            (hw + (obj.pos.x + tri.b.x) * w) as i32, 
                            (hh + (obj.pos.z + tri.b.z) * h) as i32
                        )
                    );

                    if !obj.is_mesh {
                        _ = self.canvas.draw_line(
                            Point::new(
                                (hw + (obj.pos.x + tri.b.x) * w) as i32, 
                                (hh + (obj.pos.z + tri.b.z) * h) as i32
                            ), 
                            Point::new(
                                (hw + (obj.pos.x + tri.c.x) * w) as i32, 
                                (hh + (obj.pos.z + tri.c.z) * h) as i32
                            )
                        );

                        if !obj.is_open {
                            _ = self.canvas.draw_line(
                                Point::new(
                                    (hw + (obj.pos.x + tri.c.x) * w) as i32, 
                                    (hh + (obj.pos.z + tri.c.z) * h) as i32
                                ), 
                                Point::new(
                                    (hw + (obj.pos.x + tri.a.x) * w) as i32, 
                                    (hh + (obj.pos.z + tri.a.z) * h) as i32
                                )
                            );
                        }
                    }
                }
            }
        }

        self.canvas.present();
    }
}