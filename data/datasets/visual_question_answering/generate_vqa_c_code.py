template = """
@dataset_register(
    name='VQAv2_split1_c_{}', 
    classes=all_classes[0: 100], 
    task_type='Visual Question Answering',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class VQAv2_split1_c_{}(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        if transform is None:
            transform = None
            self.transform = transform
        dataset = _VQAv2_split1_c(root_dir, split, "{}", classes, ignore_classes, idx_map)
        return dataset
"""


# for c in 'gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur, motion_blur, zoom_blur, snow, frost, fog, brightness, contrast, elastic_transform, pixelate, jpeg_compression, speckle_noise, gaussian_blur, spatter, saturate'.split(', '):
#     print(template.format(c, c, c))
#     print()
    # break
    
    
classes_name = [f'VQAv2_split1_c_{c}' for c in 'gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur, motion_blur, zoom_blur, snow, frost, fog, brightness, contrast, elastic_transform, pixelate, jpeg_compression, speckle_noise, gaussian_blur, spatter, saturate'.split(', ')]
print(', '.join(classes_name))