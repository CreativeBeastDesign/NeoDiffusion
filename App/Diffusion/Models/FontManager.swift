import Foundation
import CoreText

public struct FontManager {
    public static func registerCustomFonts() {
        let fontFiles = [
            ("Illinois_Mono_Regular", "ttf"),
            ("LexendDeca-VariableFont_wght", "ttf"),
            ("LexendExa-VariableFont_wght", "ttf")
        ]
        
        for (name, ext) in fontFiles {
            guard let url = Bundle.main.url(forResource: name, withExtension: ext) else {
                print("[FontManager] WARNING: Font \(name).\(ext) not found in main bundle resources.")
                continue
            }
            
            var errorRef: Unmanaged<CFError>?
            let success = CTFontManagerRegisterFontsForURL(url as CFURL, .process, &errorRef)
            if success {
                print("[FontManager] Successfully registered font: \(name)")
            } else {
                if let error = errorRef?.takeRetainedValue() {
                    print("[FontManager] Font \(name) registration detail: \(error)")
                }
            }
        }
    }
}
