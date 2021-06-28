use crate::error::{ClError, ClResult};
use lazy_static::lazy_static;
use log::*;
use rust_gpu_tools::opencl::{cl_device_id, Device};
use std::collections::HashMap;
use std::fmt;
use std::ptr;
use std::sync::{Arc, Mutex, RwLock};
use triton::bindings;
use triton::FutharkContext;

const MAX_LEN: usize = 128;

lazy_static! {
    static ref FUTHARK_CONTEXT_MAP: RwLock<HashMap<Device, Arc<Mutex<FutharkContext>>>> =
        RwLock::new(HashMap::new());
}

fn create_context(device: bindings::cl_device_id) -> ClResult<bindings::cl_context> {
    let mut res = 0i32;
    let context = unsafe {
        bindings::clCreateContext(
            ptr::null(),
            1,
            [device].as_mut_ptr(),
            None,
            ptr::null_mut(),
            &mut res,
        )
    };
    if res == bindings::CL_SUCCESS as i32 {
        Ok(context)
    } else {
        Err(ClError::CannotCreateContext)
    }
}

fn create_queue(
    context: bindings::cl_context,
    device: bindings::cl_device_id,
) -> ClResult<bindings::cl_command_queue> {
    let mut res = 0i32;
    let context = unsafe { bindings::clCreateCommandQueue(context, device, 0, &mut res) };
    if res == bindings::CL_SUCCESS as i32 {
        Ok(context)
    } else {
        Err(ClError::CannotCreateQueue)
    }
}

fn create_futhark_context(device: bindings::cl_device_id) -> ClResult<FutharkContext> {
    unsafe {
        let context = create_context(device)?;
        let queue = create_queue(context, device)?;

        let ctx_config = bindings::futhark_context_config_new();
        let ctx = bindings::futhark_context_new_with_command_queue(ctx_config, queue);
        Ok(FutharkContext {
            context: ctx,
            config: ctx_config,
        })
    }
}

pub fn futhark_context(device: &Device) -> ClResult<Arc<Mutex<FutharkContext>>> {
    info!("getting context for ~{:?}", device.name());
    let mut map = FUTHARK_CONTEXT_MAP.write().unwrap();

    if !map.contains_key(&device) {
        info!("device: {:?}", device);
        let cl_device_id = device.cl_device_id() as bindings::cl_device_id;
        let context = create_futhark_context(cl_device_id)?;
        map.insert(device.clone(), Arc::new(Mutex::new(context)));
    }
    Ok(Arc::clone(&map[&device]))
}

pub fn default_futhark_context() -> ClResult<Arc<Mutex<FutharkContext>>> {
    info!("getting default futhark context");
    let bus_id = std::env::var("NEPTUNE_DEFAULT_GPU")
        .ok()
        .and_then(|v| match v.parse::<u32>() {
            Ok(bus_id) => Some(bus_id),
            Err(_) => {
                error!("Bus-id '{}' is given in wrong format!", v);
                None
            }
        });
    match bus_id {
        Some(bus_id) => {
            info!(
                "Using device with bus-id {} for creating the FutharkContext...",
                bus_id
            );
            match Device::by_bus_id(bus_id) {
                Ok(device) => futhark_context(device),
                Err(_) => {
                    error!(
                       "A device with the given bus-id doesn't exist! Defaulting to the first device..."
                   );
                    let all = Device::all();
                    let device = all.first().ok_or(ClError::DeviceNotFound)?;
                    futhark_context(device)
                }
            }
        }
        None => {
            let all = Device::all();
            let device = all.first().ok_or(ClError::DeviceNotFound)?;
            futhark_context(device)
        }
    }
}

fn to_u32(inp: &[u8]) -> u32 {
    (inp[0] as u32) + ((inp[1] as u32) << 8) + ((inp[2] as u32) << 16) + ((inp[3] as u32) << 24)
}
