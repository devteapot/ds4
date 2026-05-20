#include <stdbool.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>

static void ds4_av_set_err(char *err, size_t errlen, NSString *msg) {
    if (!err || errlen == 0) return;
    const char *s = msg ? [msg UTF8String] : "video decode failed";
    snprintf(err, errlen, "%s", s ? s : "video decode failed");
}

bool ds4_av_fetch_url_bytes(const char *url,
                            unsigned char **bytes_out,
                            size_t *len_out,
                            char *err,
                            size_t errlen) {
    if (bytes_out) *bytes_out = NULL;
    if (len_out) *len_out = 0;
    if (!url || !bytes_out || !len_out) {
        ds4_av_set_err(err, errlen, @"invalid remote media URL");
        return false;
    }

    @autoreleasepool {
        NSString *url_string = [NSString stringWithUTF8String:url];
        if (!url_string) {
            ds4_av_set_err(err, errlen, @"invalid remote media URL");
            return false;
        }
        NSURLComponents *components =
            [NSURLComponents componentsWithString:url_string];
        NSString *scheme = components.scheme.lowercaseString;
        if (![scheme isEqualToString:@"http"] &&
            ![scheme isEqualToString:@"https"]) {
            ds4_av_set_err(err, errlen,
                           @"remote media URL must use http or https");
            return false;
        }
        NSURL *nsurl = components.URL ?: [NSURL URLWithString:url_string];
        if (!nsurl) {
            ds4_av_set_err(err, errlen, @"invalid remote media URL");
            return false;
        }

        NSMutableURLRequest *request =
            [NSMutableURLRequest requestWithURL:nsurl
                                    cachePolicy:NSURLRequestReloadIgnoringLocalCacheData
                                timeoutInterval:15.0];
        request.HTTPMethod = @"GET";
        [request setValue:@"ds4/1.0" forHTTPHeaderField:@"User-Agent"];

        NSURLResponse *response = nil;
        NSError *error = nil;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        NSData *payload =
            [NSURLConnection sendSynchronousRequest:request
                                  returningResponse:&response
                                              error:&error];
#pragma clang diagnostic pop
        if (!payload) {
            NSString *reason = error.localizedDescription ?:
                @"failed to fetch remote media URL";
            ds4_av_set_err(err, errlen,
                           [NSString stringWithFormat:
                               @"failed to fetch remote media URL: %@",
                               reason]);
            return false;
        }
        if ([response isKindOfClass:[NSHTTPURLResponse class]]) {
            NSInteger status = [(NSHTTPURLResponse *)response statusCode];
            if (status < 200 || status >= 300) {
                ds4_av_set_err(err, errlen,
                               [NSString stringWithFormat:
                                   @"remote media URL returned HTTP %ld",
                                   (long)status]);
                return false;
            }
        }

        static const NSUInteger max_remote_media_bytes =
            (NSUInteger)64u * 1024u * 1024u;
        NSUInteger n = payload.length;
        if (n == 0) {
            ds4_av_set_err(err, errlen,
                           @"remote media URL returned an empty body");
            return false;
        }
        if (n > max_remote_media_bytes) {
            ds4_av_set_err(err, errlen,
                           @"remote media payload exceeds 64 MiB limit");
            return false;
        }
        unsigned char *copy = malloc((size_t)n);
        if (!copy) {
            ds4_av_set_err(err, errlen,
                           @"failed to allocate remote media payload");
            return false;
        }
        memcpy(copy, payload.bytes, (size_t)n);
        *bytes_out = copy;
        *len_out = (size_t)n;
        return true;
    }
}

static bool ds4_av_copy_pixel_buffer_rgb(CVPixelBufferRef pixel,
                                         unsigned char **rgb_out,
                                         uint32_t *width_out,
                                         uint32_t *height_out,
                                         char *err,
                                         size_t errlen) {
    if (!pixel || !rgb_out || !width_out || !height_out) {
        ds4_av_set_err(err, errlen, @"invalid video pixel buffer");
        return false;
    }
    *rgb_out = NULL;
    if (width_out) *width_out = 0;
    if (height_out) *height_out = 0;

    CVReturn cv = CVPixelBufferLockBaseAddress(pixel,
                                               kCVPixelBufferLock_ReadOnly);
    if (cv != kCVReturnSuccess) {
        ds4_av_set_err(err, errlen, @"failed to lock video frame pixels");
        return false;
    }

    const size_t width = CVPixelBufferGetWidth(pixel);
    const size_t height = CVPixelBufferGetHeight(pixel);
    const size_t stride = CVPixelBufferGetBytesPerRow(pixel);
    const unsigned char *base =
        (const unsigned char *)CVPixelBufferGetBaseAddress(pixel);
    const uint64_t pixels = (uint64_t)width * (uint64_t)height;
    bool ok = base != NULL &&
              width > 0 && height > 0 &&
              width <= UINT32_MAX && height <= UINT32_MAX &&
              pixels / width == height &&
              pixels <= SIZE_MAX / 3u &&
              stride >= width * 4u;
    unsigned char *rgb = NULL;
    if (ok) {
        rgb = malloc((size_t)pixels * 3u);
        ok = rgb != NULL;
    }
    if (ok) {
        for (size_t y = 0; y < height; y++) {
            const unsigned char *src = base + y * stride;
            for (size_t x = 0; x < width; x++) {
                const unsigned char *px = src + x * 4u;
                uint64_t dst = ((uint64_t)y * width + x) * 3u;
                rgb[dst + 0u] = px[2];
                rgb[dst + 1u] = px[1];
                rgb[dst + 2u] = px[0];
            }
        }
        *rgb_out = rgb;
        *width_out = (uint32_t)width;
        *height_out = (uint32_t)height;
        rgb = NULL;
    } else {
        ds4_av_set_err(err, errlen,
                       @"failed to copy video frame RGB pixels");
    }
    free(rgb);
    CVPixelBufferUnlockBaseAddress(pixel, kCVPixelBufferLock_ReadOnly);
    return ok;
}

static void ds4_av_free_frames(unsigned char **frames,
                               uint32_t *widths,
                               uint32_t *heights,
                               uint32_t count) {
    if (frames) {
        for (uint32_t i = 0; i < count; i++) free(frames[i]);
    }
    free(frames);
    free(widths);
    free(heights);
}

enum {
    DS4_AV_QWEN_FRAME_FACTOR = 2u,
    DS4_AV_QWEN_MAX_VIDEO_FRAMES = 256u,
};

static uint32_t ds4_av_round_by_factor_u32(uint32_t n, uint32_t factor) {
    if (factor == 0) return n;
    return ((n + factor / 2u) / factor) * factor;
}

static uint32_t ds4_av_ceil_by_factor_u32(uint32_t n, uint32_t factor) {
    if (factor == 0) return n;
    return ((n + factor - 1u) / factor) * factor;
}

static uint32_t ds4_av_floor_by_factor_u32(uint32_t n, uint32_t factor) {
    if (factor == 0) return n;
    return (n / factor) * factor;
}

static uint32_t ds4_av_qwen_frame_count(uint32_t range_frames,
                                        uint32_t nframes,
                                        uint32_t min_frames,
                                        uint32_t max_frames,
                                        bool has_fps,
                                        double fps,
                                        double source_fps) {
    if (range_frames == 0) return 0;
    if (range_frames == 1) return 1;

    uint32_t min_aligned = 0;
    if (min_frames > 0) {
        min_aligned = ds4_av_ceil_by_factor_u32(
            min_frames, DS4_AV_QWEN_FRAME_FACTOR);
    } else if (has_fps) {
        min_aligned = 4;
    }

    uint32_t max_aligned = 0;
    if (max_frames > 0) {
        max_aligned = ds4_av_floor_by_factor_u32(
            max_frames, DS4_AV_QWEN_FRAME_FACTOR);
        if (max_aligned < DS4_AV_QWEN_FRAME_FACTOR) {
            max_aligned = DS4_AV_QWEN_FRAME_FACTOR;
        }
    }

    uint32_t target = 0;
    if (nframes > 0) {
        target = ds4_av_round_by_factor_u32(nframes,
                                            DS4_AV_QWEN_FRAME_FACTOR);
    } else if (has_fps && fps > 0.0 && source_fps > 0.0) {
        double sampled = ((double)range_frames / source_fps) * fps;
        if (!isfinite(sampled) || sampled < 1.0) sampled = 1.0;
        if (sampled > (double)UINT32_MAX) sampled = (double)UINT32_MAX;
        target = (uint32_t)floor(sampled);
    } else {
        target = range_frames;
    }

    if (min_aligned > 0 && target < min_aligned) target = min_aligned;
    if (max_aligned > 0 && target > max_aligned) target = max_aligned;
    if (target > range_frames) {
        target = ds4_av_floor_by_factor_u32(range_frames,
                                            DS4_AV_QWEN_FRAME_FACTOR);
    }
    if (target > DS4_AV_QWEN_MAX_VIDEO_FRAMES) {
        target = DS4_AV_QWEN_MAX_VIDEO_FRAMES;
    }
    target = ds4_av_floor_by_factor_u32(target, DS4_AV_QWEN_FRAME_FACTOR);
    if (target < DS4_AV_QWEN_FRAME_FACTOR) target = DS4_AV_QWEN_FRAME_FACTOR;
    if (target > range_frames) {
        target = ds4_av_floor_by_factor_u32(range_frames,
                                            DS4_AV_QWEN_FRAME_FACTOR);
    }
    if (target < DS4_AV_QWEN_FRAME_FACTOR) target = DS4_AV_QWEN_FRAME_FACTOR;
    return target;
}

bool ds4_av_decode_video_frames_rgb(const unsigned char *data,
                                    size_t len,
                                    uint32_t nframes,
                                    uint32_t min_frames,
                                    uint32_t max_frames,
                                    bool has_fps,
                                    double fps,
                                    bool has_video_start,
                                    double video_start,
                                    bool has_video_end,
                                    double video_end,
                                    unsigned char ***frames_out,
                                    uint32_t **widths_out,
                                    uint32_t **heights_out,
                                    uint32_t *count_out,
                                    char *err,
                                    size_t errlen) {
    if (frames_out) *frames_out = NULL;
    if (widths_out) *widths_out = NULL;
    if (heights_out) *heights_out = NULL;
    if (count_out) *count_out = 0;
    if (!data || len == 0 || !frames_out || !widths_out ||
        !heights_out || !count_out) {
        ds4_av_set_err(err, errlen, @"invalid video frame decode request");
        return false;
    }
    if (nframes > 0 && max_frames < nframes) max_frames = nframes;
    if (max_frames == 0) max_frames = 4;
    if (max_frames > DS4_AV_QWEN_MAX_VIDEO_FRAMES) {
        max_frames = DS4_AV_QWEN_MAX_VIDEO_FRAMES;
    }

    @autoreleasepool {
        NSData *payload = [NSData dataWithBytes:data length:len];
        if (!payload) {
            ds4_av_set_err(err, errlen, @"failed to create video payload");
            return false;
        }

        NSString *name = [[NSUUID UUID].UUIDString stringByAppendingPathExtension:@"mp4"];
        NSString *path = [NSTemporaryDirectory() stringByAppendingPathComponent:name];
        if (![payload writeToFile:path atomically:YES]) {
            ds4_av_set_err(err, errlen, @"failed to stage video payload");
            return false;
        }

        NSURL *url = [NSURL fileURLWithPath:path];
        AVURLAsset *asset = [AVURLAsset URLAssetWithURL:url options:nil];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        NSArray<AVAssetTrack *> *tracks =
            [asset tracksWithMediaType:AVMediaTypeVideo];
#pragma clang diagnostic pop
        AVAssetTrack *track = tracks.firstObject;
        if (!track) {
            [[NSFileManager defaultManager] removeItemAtPath:path error:nil];
            ds4_av_set_err(err, errlen, @"video payload has no video track");
            return false;
        }

        NSDictionary *settings = @{
            (id)kCVPixelBufferPixelFormatTypeKey:
                @(kCVPixelFormatType_32BGRA)
        };

        NSError *countError = nil;
        AVAssetReader *countReader =
            [[AVAssetReader alloc] initWithAsset:asset error:&countError];
        if (!countReader) {
            [[NSFileManager defaultManager] removeItemAtPath:path error:nil];
            NSString *msg = countError.localizedDescription ?:
                @"failed to create video frame counter";
            ds4_av_set_err(err, errlen, msg);
            return false;
        }
        AVAssetReaderTrackOutput *countOutput =
            [[AVAssetReaderTrackOutput alloc] initWithTrack:track
                                             outputSettings:settings];
        countOutput.alwaysCopiesSampleData = NO;
        if (![countReader canAddOutput:countOutput]) {
            [[NSFileManager defaultManager] removeItemAtPath:path error:nil];
            ds4_av_set_err(err, errlen,
                           @"video reader cannot count RGB frames");
            return false;
        }
        [countReader addOutput:countOutput];
        if (![countReader startReading]) {
            [[NSFileManager defaultManager] removeItemAtPath:path error:nil];
            NSString *msg = countReader.error.localizedDescription ?:
                @"failed to start video frame counter";
            ds4_av_set_err(err, errlen, msg);
            return false;
        }
        uint32_t total_frames = 0;
        for (;;) {
            CMSampleBufferRef sample = [countOutput copyNextSampleBuffer];
            if (!sample) break;
            CFRelease(sample);
            if (total_frames != UINT32_MAX) total_frames++;
        }
        if (total_frames == 0) {
            [[NSFileManager defaultManager] removeItemAtPath:path error:nil];
            NSString *msg = countReader.error.localizedDescription ?:
                @"failed to count video frames";
            ds4_av_set_err(err, errlen, msg);
            return false;
        }

        double source_fps = (double)track.nominalFrameRate;
        if (!isfinite(source_fps) || source_fps <= 0.0) {
            double seconds = CMTimeGetSeconds(asset.duration);
            if (isfinite(seconds) && seconds > 0.0) {
                source_fps = (double)total_frames / seconds;
            }
        }
        if (!isfinite(source_fps) || source_fps <= 0.0) source_fps = 30.0;

        uint32_t start_frame = 0;
        uint32_t end_frame = total_frames - 1u;
        double duration_seconds = (double)total_frames / source_fps;
        if (has_video_start) {
            double s = video_start;
            if (!isfinite(s) || s < 0.0) s = 0.0;
            if (s > duration_seconds) s = duration_seconds;
            double idx = ceil(s * source_fps);
            if (!isfinite(idx) || idx < 0.0) idx = 0.0;
            if (idx > (double)(total_frames - 1u)) {
                idx = (double)(total_frames - 1u);
            }
            start_frame = (uint32_t)idx;
        }
        if (has_video_end) {
            double e = video_end;
            if (!isfinite(e) || e < 0.0) e = 0.0;
            if (e > duration_seconds) e = duration_seconds;
            double idx = floor(e * source_fps);
            if (!isfinite(idx) || idx < 0.0) idx = 0.0;
            if (idx > (double)(total_frames - 1u)) {
                idx = (double)(total_frames - 1u);
            }
            end_frame = (uint32_t)idx;
        }
        if (start_frame > end_frame) {
            [[NSFileManager defaultManager] removeItemAtPath:path error:nil];
            ds4_av_set_err(err, errlen,
                           @"invalid video_start/video_end frame range");
            return false;
        }
        uint32_t range_frames = end_frame - start_frame + 1u;

        NSError *error = nil;
        AVAssetReader *reader =
            [[AVAssetReader alloc] initWithAsset:asset error:&error];
        if (!reader) {
            [[NSFileManager defaultManager] removeItemAtPath:path error:nil];
            NSString *msg = error.localizedDescription ?:
                @"failed to create video reader";
            ds4_av_set_err(err, errlen, msg);
            return false;
        }
        AVAssetReaderTrackOutput *output =
            [[AVAssetReaderTrackOutput alloc] initWithTrack:track
                                             outputSettings:settings];
        output.alwaysCopiesSampleData = NO;
        if (![reader canAddOutput:output]) {
            [[NSFileManager defaultManager] removeItemAtPath:path error:nil];
            ds4_av_set_err(err, errlen,
                           @"video reader cannot output RGB frames");
            return false;
        }
        [reader addOutput:output];
        if (![reader startReading]) {
            [[NSFileManager defaultManager] removeItemAtPath:path error:nil];
            NSString *msg = reader.error.localizedDescription ?:
                @"failed to start video reader";
            ds4_av_set_err(err, errlen, msg);
            return false;
        }

        uint32_t target_count = ds4_av_qwen_frame_count(
            range_frames, nframes, min_frames, max_frames, has_fps, fps,
            source_fps);
        if (target_count == 0) target_count = 1;
        unsigned char **frames = calloc(target_count, sizeof(frames[0]));
        uint32_t *widths = calloc(target_count, sizeof(widths[0]));
        uint32_t *heights = calloc(target_count, sizeof(heights[0]));
        uint32_t *targets = calloc(target_count, sizeof(targets[0]));
        if (!frames || !widths || !heights || !targets) {
            [[NSFileManager defaultManager] removeItemAtPath:path error:nil];
            ds4_av_free_frames(frames, widths, heights, target_count);
            free(targets);
            ds4_av_set_err(err, errlen,
                           @"failed to allocate video frame list");
            return false;
        }
        for (uint32_t i = 0; i < target_count; i++) {
            uint32_t idx = start_frame;
            if (target_count > 1) {
                double pos = (double)start_frame +
                             (double)i * (double)(end_frame - start_frame) /
                             (double)(target_count - 1u);
                double rounded = round(pos);
                if (!isfinite(rounded) || rounded < (double)start_frame) {
                    rounded = (double)start_frame;
                }
                if (rounded > (double)end_frame) {
                    rounded = (double)end_frame;
                }
                idx = (uint32_t)rounded;
            }
            targets[i] = idx;
        }

        uint32_t count = 0;
        uint32_t frame_index = 0;
        while (count < target_count) {
            CMSampleBufferRef sample = [output copyNextSampleBuffer];
            if (!sample) break;
            bool keep = frame_index == targets[count];
            frame_index++;
            if (!keep) {
                CFRelease(sample);
                continue;
            }
            CVImageBufferRef pixel = CMSampleBufferGetImageBuffer(sample);
            unsigned char *rgb = NULL;
            uint32_t width = 0;
            uint32_t height = 0;
            bool copied = ds4_av_copy_pixel_buffer_rgb(
                pixel, &rgb, &width, &height, err, errlen);
            CFRelease(sample);
            if (!copied) {
                ds4_av_free_frames(frames, widths, heights, count);
                free(targets);
                [[NSFileManager defaultManager] removeItemAtPath:path error:nil];
                return false;
            }
            frames[count] = rgb;
            widths[count] = width;
            heights[count] = height;
            count++;
        }
        [[NSFileManager defaultManager] removeItemAtPath:path error:nil];

        if (count == 0) {
            NSString *msg = reader.error.localizedDescription ?:
                @"failed to decode video frames";
            ds4_av_free_frames(frames, widths, heights, count);
            free(targets);
            ds4_av_set_err(err, errlen, msg);
            return false;
        }
        free(targets);
        *frames_out = frames;
        *widths_out = widths;
        *heights_out = heights;
        *count_out = count;
        return true;
    }
}

bool ds4_av_decode_video_first_frame_rgb(const unsigned char *data,
                                         size_t len,
                                         unsigned char **rgb_out,
                                         uint32_t *width_out,
                                         uint32_t *height_out,
                                         char *err,
                                         size_t errlen) {
    if (rgb_out) *rgb_out = NULL;
    if (width_out) *width_out = 0;
    if (height_out) *height_out = 0;
    unsigned char **frames = NULL;
    uint32_t *widths = NULL;
    uint32_t *heights = NULL;
    uint32_t count = 0;
    if (!ds4_av_decode_video_frames_rgb(data, len, 1, 0, 1,
                                        false, 0.0,
                                        false, 0.0,
                                        false, 0.0,
                                        &frames, &widths,
                                        &heights, &count, err, errlen)) {
        return false;
    }
    if (count == 0 || !frames || !widths || !heights) {
        ds4_av_free_frames(frames, widths, heights, count);
        ds4_av_set_err(err, errlen, @"failed to decode video first frame");
        return false;
    }
    *rgb_out = frames[0];
    *width_out = widths[0];
    *height_out = heights[0];
    frames[0] = NULL;
    ds4_av_free_frames(frames, widths, heights, count);
    return true;
}
